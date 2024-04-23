import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import operator
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from collections import Counter

class EncoderCNN(nn.Module):
    def __init__(self, target_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features
    
class EncoderYOLO(nn.Module):

    def __init__(self, target_size):
        super(EncoderYOLO, self).__init__()

        self.yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        for param in self.yolo.model.parameters():
            param.requires_grad = False

        # register activation to store 3 feature maps
        self.activation = {}
        def get_input():
            def hook(model, input, output):
                data = input[0]
                self.activation["large"] = data[0].detach()
                self.activation["medium"] = data[1].detach()
                self.activation["small"] = data[2].detach()
            return hook
        self.yolo.model.model.model[24].register_forward_hook(get_input())

        self.linear = nn.Linear(3 * 3 * 7 * 7 * 85, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        return
    
    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def forward(self, images):
        _ = self.yolo(images)

        small_act = torch.clone(self.activation["small"]) # (B, 3, 8, 8, 85)
        medium_act = torch.clone(self.activation["medium"])
        large_act = torch.clone(self.activation["large"])

        B, _, _, _, _ = small_act.shape
        small_act = small_act.reshape(B, -1)

        B, C, H, W, L = medium_act.shape
        medium_act = torch.permute(medium_act, (0, 1, 4, 2, 3)).reshape((B, C*L, H, W))
        medium_act = self.pool(medium_act)
        medium_act = medium_act.reshape(B, -1)

        B, C, H, W, L = large_act.shape
        large_act= torch.permute(large_act, (0, 1, 4, 2, 3)).reshape((B, C*L, H, W))
        large_act= self.pool(large_act)
        large_act= self.pool(large_act)
        large_act = large_act.reshape(B, -1)

        # concatenate all activations for encoding
        act = torch.concat([small_act, medium_act, large_act], dim=1)
        act = self.linear(act)
        act = self.bn(act)
        return act


class EncoderStory(nn.Module):
    def __init__(self, img_feature_size=1024, hidden_size=1024, n_layers=2):
        super(EncoderStory, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # self.cnn = EncoderCNN(img_feature_size)
        self.cnn = EncoderYOLO(img_feature_size)

        self.lstm = nn.LSTM(img_feature_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=0.5)

        encoder_layer = nn.TransformerEncoderLayer(d_model=img_feature_size, nhead=4)
        layer_norm = nn.LayerNorm(img_feature_size)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=layer_norm
        )
        self.transformer_linear = nn.Linear(
            in_features=img_feature_size,
            out_features=2*hidden_size
        )

        self.linear = nn.Linear(hidden_size * 2 + img_feature_size, hidden_size * 2)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(hidden_size * 2, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return self.cnn.get_params() + list(self.lstm.parameters()) + list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_images):
        data_size = story_images.size()
        local_cnn = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4]))   #  local_cnn: (B * 5, img_emb_dim)
        local_cnn = local_cnn.view(data_size[0], data_size[1], -1)

        transformer_out = self.transformer(local_cnn)
        global_transformer = self.transformer_linear(transformer_out)

        glocal = torch.cat((local_cnn, global_transformer), 2)

        output = self.linear(glocal)
        output = self.dropout(output)
        output = self.bn(output.contiguous().view(-1, self.hidden_size * 2)).view(data_size[0], data_size[1], -1)

        return output, (None, None)


class DecoderStory(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab):
        super(DecoderStory, self).__init__()

        self.embed_size = embed_size
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.rnn = DecoderRNN(embed_size, hidden_size, 2, vocab)
        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_feature, captions, lengths):
        '''
            story_features: (5 x img_emb_dim)
            captions: (5 x max_seq_len), max_seq_len = max(lengths)
            lengths: list of lengths of each caption
        '''
        story_feature = self.linear(story_feature)
        story_feature = self.dropout(story_feature)
        story_feature = F.relu(story_feature)
        result = self.rnn(story_feature, captions, lengths)
        return result

    def inference(self, story_feature):
        story_feature = self.linear(story_feature)
        story_feature = F.relu(story_feature)
        result = self.rnn.inference(story_feature)
        return result


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, vocab):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        vocab_size = len(vocab)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout1 = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(0)

        self.brobs = []

        self.init_input = torch.zeros([5, 1, embed_size], dtype=torch.float32)

        if torch.cuda.is_available():
            self.init_input = self.init_input.cuda()

        self.start_vec = torch.zeros([1, vocab_size], dtype=torch.float32)
        self.start_vec[0][1] = 10000
        if torch.cuda.is_available():
            self.start_vec = self.start_vec.cuda()

        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_hidden(self):
        h0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)
        c0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)

        h0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)
        c0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)
        
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
            
        return (h0, c0)

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0) 

    def forward(self, features, captions, lengths):
        '''
            story_features: (5 x img_emb_dim)
            captions: (5 x max_seq_len), max_seq_len = max(lengths)
            lengths: list of lengths of each caption
        '''

        embeddings = self.embed(captions)       # (5, max_seq_len, 256)
        embeddings = self.dropout1(embeddings)  # (5, max_seq_len, 256)
        features = features.unsqueeze(1).expand(-1, np.amax(lengths), -1) # (5, max_seq_len, img_emb_dim)
        embeddings = torch.cat((features, embeddings), 2)   # (5, max_seq_len, 256 + img_emb_dim)

        outputs = []
        (hn, cn) = self.init_hidden()

        for i, length in enumerate(lengths):
            lstm_input = embeddings[i][0:length - 1]
            output, (hn, cn) = self.lstm(lstm_input.unsqueeze(0), (hn, cn))
            output = self.dropout2(output)
            output = self.linear(output[0])
            output = torch.cat((self.start_vec, output), 0)
            outputs.append(output)

        return outputs


    def inference(self, features):
        results = []
        (hn, cn) = self.init_hidden()
        vocab = self.vocab
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'), vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'), vocab('did')]

        cumulated_word = []
        for feature in features:

            feature = feature.unsqueeze(0).unsqueeze(0)
            predicted = torch.tensor([1], dtype=torch.long).cuda()
            lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)
            sampled_ids = [predicted,]

            count = 0
            prob_sum = 1.0

            for i in range(50):
                outputs, (hn, cn) = self.lstm(lstm_input, (hn, cn))
                outputs = self.linear(outputs.squeeze(1))

                if predicted not in termination_list:
                    outputs[0][end_vocab] = -100.0

                for forbidden in forbidden_list:
                    outputs[0][forbidden] = -100.0

                cumulated_counter = Counter()
                cumulated_counter.update(cumulated_word)

                prob_res = outputs[0]
                prob_res = self.softmax(prob_res)
                for word, cnt in cumulated_counter.items():
                    if cnt > 0 and word not in function_list:
                        prob_res[word] = prob_res[word] / (1.0 + cnt * 5.0)
                prob_res = prob_res * (1.0 / prob_res.sum())

                candidate = []
                for i in range(100):
                    index = np.random.choice(prob_res.size()[0], 1, p=prob_res.cpu().detach().numpy())[0]
                    candidate.append(index)

                counter = Counter()
                counter.update(candidate)

                sorted_candidate = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)

                predicted, _ = counter.most_common(1)[0]
                cumulated_word.append(predicted)

                predicted = torch.from_numpy(np.array([predicted])).cuda()
                sampled_ids.append(predicted)

                if predicted == 2:
                    break

                lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)

            results.append(sampled_ids)

        return results
