import os
import re
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def get_parser():
    parser = argparse.ArgumentParser(description='Datagrand Parser')
    parser.add_argument('--train-data', default='train.txt', help='path of training data')
    parser.add_argument('--test-data', default='test.txt', help='path of test data')
    parser.add_argument('--train-data-no-tags', default='train_no_tags.txt', help='path of training data with no tags')
    parser.add_argument('--model', default='crf', help='model name')
    parser.add_argument('-f', type=int, default=3, help='hyper-parameter f of crf model')
    parser.add_argument('-c', type=int, default=1, help='hyper-parameter f of crf model')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--split-valid', type=float, default=0.2, help='random seed')
    return parser


def remove_tags(data_path, save_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        results = []
        for line in lines:
            line = re.sub('/[abco] +', '_', line)
            line = re.sub('/[abco]\n', '\n', line)
            results.append(line)
    with open(save_path, 'w', encoding='utf-8') as f_out:
        f_out.writelines(results)


def make_data_with_tags(data, split='train'):
    results = []
    for line in data:
        features = []
        tags = []
        samples = line.strip().split('  ')
        for sample in samples:
            sample_list = sample[:-2].split('_')
            tag = sample[-1]
            features.extend(sample_list)
            tags.extend(['O'] * len(sample_list)) if tag == 'o' else tags.extend(['B-' + tag] + ['I-' + tag] * (len(sample_list)-1))
        results.append({'features': features, 'tags': tags})
    train_write_list = []
    with open('dg_'+split+'.txt', 'w', encoding='utf-8') as f_out:
        for result in results:
            for i in range(len(result['tags'])):
                train_write_list.append(result['features'][i] + '\t' + result['tags'][i] + '\n')
            train_write_list.append('\n')
        f_out.writelines(train_write_list)


def make_data_with_no_tags(data, split='valid'):
    results = []
    for line in data:
        features = []
        sample_list = line.strip().split('_')
        features.extend(sample_list)
        results.append({'features': features})
    test_write_list = []
    with open('dg_'+split+'.txt', 'w', encoding='utf-8') as f_out:
        for result in results:
            for i in range(len(result['features'])):
                test_write_list.append(result['features'][i] + '\n')
            test_write_list.append('\n')
        f_out.writelines(test_write_list)


def crf_train(f, c):
    crf_train = "crf_learn -f %d -c %d template.txt dg_train.txt dg_model"%(f, c)
    os.system(crf_train)


def crf_generate(split='valid_no_tags'):
    if split == 'train_no_tags':
        cmd = "crf_test -m dg_model dg_train_no_tags.txt -o dg_train_result.txt" 
    elif split == 'valid_no_tags':
        cmd = "crf_test -m dg_model dg_valid_no_tags.txt -o dg_valid_result.txt"
    else:
        cmd = "crf_test -m dg_model dg_test.txt -o dg_result.txt"
    os.system(cmd)


def evaluate(predictions, true_data):
    with open(predictions, 'r', encoding='utf-8') as f:
        pred = f.readlines()
    with open(true_data, 'r', encoding='utf-8') as f_true:
        true = f_true.readlines()
    pred_tags = []
    true_tags = []
    for line in pred:
        if line == '' or line == '\n':
            continue
        tag = re.sub('[IB]-', '', line.split()[-1])
        pred_tags.append(tag)
    for line in true:
        if line == '' or line == '\n':
            continue
        tag = re.sub('[IB]-', '', line.split()[-1])
        true_tags.append(tag)
    assert len(true_tags) == len(pred_tags)

    correct = 0
    total_pred = pred_tags.count('a') + pred_tags.count('b') + pred_tags.count('c')

    total_true = true_tags.count('a') + true_tags.count('b') + true_tags.count('c')
    for pred, true in zip(pred_tags, true_tags):
        if pred == true and pred in ['a', 'b', 'c']:
            correct += 1

    def micro_f1(correct, num_pred, num_true):
        p = correct / num_pred
        r = correct / num_true
        f1_score = 2 * p * r / (p + r)
        return (p, r, f1_score)

    p, r, f1_score = micro_f1(correct, total_pred, total_true)
    print('total correct: %d, total prediciton num: %d, total true num: %d'%(correct, total_pred, total_true))
    print('precition: %.3f, recall: %.3f, F1 score: %.3f'%(p, r, f1_score))

    return f1_score


def make_submit_data():
    f_write = open('dg_submit.txt', 'w', encoding='utf-8') 
    with open('dg_result.txt', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n')
        for line in lines:
            if line == '':
                continue
            tokens = line.split('\n')
            features = []
            tags = []
            for token in tokens:
                feature_tag = token.split()
                features.append(feature_tag[0])
                tags.append(feature_tag[-1])
            samples = []
            i = 0
            while i < len(features):
                sample = []
                if tags[i] == 'O':
                    sample.append(features[i])
                    j = i + 1
                    while j < len(features) and tags[j] == 'O':
                        sample.append(features[j])
                        j += 1
                    samples.append('_'.join(sample) + '/o')
                else:
                    if tags[i][0] != 'B':
                        print(tags[i][0] + ' error start')
                        j = i + 1
                    else:
                        sample.append(features[i])
                        j = i + 1
                        while j < len(features) and tags[j][0] == 'I' and tags[j][-1] == tags[i][-1]:
                            sample.append(features[j])
                            j += 1
                        samples.append('_'.join(sample) + '/' + tags[i][-1])
                i = j
            f_write.write('  '.join(samples) + '\n')


def preprocess(args):
    with open(args.train_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
    with open(args.train_data_no_tags, 'r', encoding='utf-8') as f:
        data_no_tags = f.readlines()
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    train_data_with_tags, valid_data_with_tags, train_data_no_tags, valid_data_no_tags = train_test_split(data, data_no_tags,
                                                                    test_size=args.split_valid, random_state=args.seed)
    make_data_with_tags(train_data_with_tags, 'train')
    make_data_with_tags(valid_data_with_tags, 'valid')
    make_data_with_no_tags(train_data_no_tags, 'train_no_tags')
    make_data_with_no_tags(valid_data_no_tags, 'valid_no_tags')
    make_data_with_no_tags(test_data, 'test')


def main(args):
    if not os.path.exists(args.train_data_no_tags):
        remove_tags(args.train_data, args.train_data_no_tags)
    preprocess(args)

    if args.model == 'crf':
        crf_train(args.f, args.c)
        crf_generate('train_no_tags')
        train_f1_score = evaluate('dg_train_result.txt', 'dg_train.txt')
        print('Training F1 score is: %.5f'%train_f1_score)
        crf_generate('valid_no_tags')
        valid_f1_score = evaluate('dg_valid_result.txt', 'dg_valid.txt')
        print('Validation F1 score is: %.5f'%valid_f1_score)
        crf_generate('test')
        make_submit_data()
        

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)