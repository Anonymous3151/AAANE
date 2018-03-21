import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from classify import Classifier, read_node_label
from graph import *
from grarep import GraRep
import time
from tqdm import tqdm
from optimizers import get_optimizer
import logging
import keras.backend as K
from keras.layers import Dense, Activation, Input, noise
from keras.models import Model
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin, Square_loss


logging.basicConfig(
                    #filename='out.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument('--input', required=True,
                    help='Input graph file')
parser.add_argument('--output', required=True,
                    help='Output representation file')
parser.add_argument('--number-walks', default=10, type=int,
                    help='Number of random walks to start at each node')
parser.add_argument('--directed', action='store_true',
                    help='Treat graph as directed.')
parser.add_argument('--walk-length', default=80, type=int,
                    help='Length of the random walk started at each node')
parser.add_argument('--workers', default=8, type=int,
                    help='Number of parallel processes.')
parser.add_argument('--representation-size', default=8, type=int,
                    help='Number of latent dimensions to learn for each node.')
parser.add_argument('--window-size', default=10, type=int,
                    help='Window size of skipgram model.')
parser.add_argument('--p', default=1.0, type=float)
parser.add_argument('--q', default=1.0, type=float)
parser.add_argument('--method', required=True, choices=['node2vec', 'deepWalk', 'line', 'gcn', 'grarep', 'tadw'],
                    help='The learning method')
parser.add_argument('--label-file', default='',
                    help='The file of node label')
parser.add_argument('--feature-file', default='',
                    help='The file of node features')
parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                    help='Input graph format')
parser.add_argument('--weighted', action='store_true',
                    help='Treat graph as weighted')
parser.add_argument('--clf-ratio', default=0.3, type=float,
                    help='The ratio of training data in the classification')
parser.add_argument('--no-auto-stop', action='store_true',
                    help='no early stop when training LINE')
parser.add_argument('--dropout', default=0.5, type=float,
                    help='Dropout rate (1 - keep probability)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='Weight for L2 loss on embedding matrix')
parser.add_argument('--hidden', default=16, type=int,
                    help='Number of units in hidden layer 1')
parser.add_argument('--kstep', default=8, type=int,
                    help='Use k-step transition probability matrix')
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200,
                    help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50,
                    help="Batch size (default=50)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=128,
                    help="The number of aspects specified by users (default=14)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=24,
                    help="Number of epochs (default=15)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=7,
                    help="Number of negative instances (default=20)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234,
                    help="Random seed (default=1234)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam',
                    help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument('--hidden_layers', dest="hidden_layers", default=3, type=int,
                    help='AutoEnocoder Layers')
parser.add_argument('--neurons_hiddenlayer', dest="neurons_hiddenlayer", default=[128, 64, 32], type=list,
                    help='Number of Neurons AE.')



args = parser.parse_args()
assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.domain in {'restaurant', 'beer'}

if args.seed > 0:
    np.random.seed(args.seed)
###############################################################################################################################
## Read Data
X, Y = read_node_label(args.label_file)
node_size = len(X)

###############################################################################################################################
## Building model

sentence_input = Input(shape=(args.kstep, node_size), dtype='float32', name='sentence_inputt')
neg_input = Input(shape=(args.neg_size, args.kstep, node_size), dtype='float32', name='neg_inputt')
predict = Input(shape=(17,), dtype='int32', name='predictt')

e_w = sentence_input
y_s = Average()(sentence_input)

att_weights = Attention(name='att_weights')([e_w, y_s])
z_s = WeightedSum()([e_w, att_weights])

##### Compute representations of negative instances #####
e_neg = neg_input
z_n = Average()(e_neg)

##### Reconstruction2 #####
dense1 = Dense(512, activation='relu')(z_s)
#dense1 = noise.GaussianNoise(0.09)(dense1)
p_t = Dense(128, activation='relu')(dense1)
dense2 = Dense(512, activation='relu')(p_t)
new_p_t = Activation('softmax', name='p_t')(dense2)
r_s = Dense(node_size, init='uniform')(new_p_t)


##### Loss1 #####

lable_ls1 = Dense(22, activation='relu')(p_t)
lable_ls = Dense(17, activation='softmax', name='lable_classification')(lable_ls1)

#loss2 = Square_loss(name='square_loss')([predict, lable_ls])
##### Loss2 #####
loss1 = MaxMargin(name='max_margin')([z_s, z_n, r_s])
##### Model #####
# model_auto = Model(input=[sentence_input, neg_input], output=[lable_ls, loss1])


#model_auto = Model(input=sentence_input, output=lable_ls)
model_auto = Model(input=[sentence_input, neg_input], output=loss1)
model_lable = Model(input=sentence_input, output=lable_ls)
low_encoder = Model(sentence_input, p_t)

###############################################################################################################################
## Training1 GRArep
t1 = time.time()
g = Graph()
print "Reading data..."
if args.graph_format == 'adjlist':
    g.read_adjlist(filename=args.input)
elif args.graph_format == 'edgelist':
    g.read_edgelist(filename=args.input, weighted=args.weighted, directed=args.directed)

model = GraRep(graph=g, Kstep=args.kstep)
vectors = model.vectors
X, Y = read_node_label(args.label_file)
node_size = len(vectors)
train_x = np.array([vectors[x] for x in X])
print("train_x.shape", train_x.shape)

# ###############################################################################################################################
# ## Prepare data

def sentence_batch_generator(data, batch_size):
    n_batch = len(data) / batch_size
    batch_count = 0
    np.random.shuffle(data)

    while True:
        if batch_count == n_batch:
            np.random.shuffle(data)
            batch_count = 0

        batch = data[batch_count*batch_size: (batch_count+1)*batch_size]
        batch_count += 1
        yield batch


def sentence_batch_generator2(data, lable,  batch_size):
    n_batch = len(data) / batch_size
    batch_count = 0
    index = np.random.permutation(node_size)
    data = data[index]
    lable = lable[index]


    while True:
        if batch_count == n_batch:
            index = np.random.permutation(node_size)
            data = data[index]
            lable = lable[index]
            batch_count = 0

        batch_data = data[batch_count*batch_size: (batch_count+1)*batch_size]
        batch_lable = lable[batch_count*batch_size: (batch_count+1)*batch_size]

        batch = np.concatenate((batch_data, batch_lable), axis=1)
        batch_count += 1
        yield batch
def negative_batch_generator(data, batch_size, neg_size):
    data_len = data.shape[0]
    dim = data.shape[1]

    while True:
        indices = np.random.choice(data_len, batch_size * neg_size)
        samples = data[indices].reshape(batch_size, neg_size, dim)
        yield samples
###############################################################################################################################
## Optimizaer algorithm
#
optimizer = get_optimizer(args)
###############################################################################################################################
## Building model
## Training2
#

logger.info('  Building model')

def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)

# model_auto.compile(optimizer=optimizer, loss=max_margin_loss, metrics=['accuracy'])
model_auto.compile(optimizer=optimizer, loss=max_margin_loss, metrics=['accuracy'])
model_lable.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model_auto.compile(optimizer=optimizer, loss={'square_loss':max_margin_loss, 'max_margin':max_margin_loss},  loss_weights=[1., 0.2])
# model_auto.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# model_auto.compile(optimizer='rmsprop',
#               loss={'lable_classification': 'categorical_crossentropy', 'max_margin': max_margin_loss},
#               loss_weights={'lable_classification': 1., 'max_margin': 0.2}, metrics=['accuracy'])
#metrics=max_margin_loss,


###############################################################################################################################
## model fit

# lable_Y = np.array(Y)
# data_len = train_x.shape[0]
# dim = train_x.shape[1]
#
#
# indices = np.random.choice(data_len, data_len * args.neg_size)
# neg_input_data = train_x[indices].reshape(data_len, args.neg_size, dim)
# lable_Y = lable_Y[indices]
#
# sen_input_data = train_x.reshape((data_len, args.kstep, node_size))
# neg_input_data = neg_input_data.reshape((data_len, args.neg_size, args.kstep, node_size))

# model_auto.fit({'sentence_inputt': sen_input_data, 'neg_inputt': neg_input_data},
#           {'max_margin': np.ones((data_len, 1))},
#                nb_epoch=3, batch_size=50)


# # ###############################################################################################################################
# # ## model fit
#
#
# lable_data = lable_Y[0:50]
#
#
#
# one_hot_lable = np.zeros((50, 17))
# for i in range(lable_data.shape[0]):
#     x = lable_data[i]
#     one_hot_lable[i, x.astype(int)] = 1
#
# model_lable.fit({'sentence_inputt': sen_input[0:50]},
#           {'lable_classification': one_hot_lable},
#                nb_epoch=15, batch_size=32)




###############################################################################################################################
## model_train_on_a_batch
lable_Y = np.array(Y)

print("type(Y)", type(Y))
print("type(np.array(Y))", type(np.array(Y)))
print("tran_x", train_x.shape, type(train_x))
print("lable_y", lable_Y.shape, type(lable_Y))
#sen_gen2 = sentence_batch_generator2(train_x, lable_Y, args.batch_size)
sen_gen = sentence_batch_generator(train_x, args.batch_size)
neg_gen = negative_batch_generator(train_x, args.batch_size, args.neg_size)
batches_per_epoch = len(train_x) / args.batch_size
print("batches_per_epoch", batches_per_epoch)
min_loss = float('inf')

lable_Y = np.array(Y)
data_len = train_x.shape[0]
dim = train_x.shape[1]

input_lable = train_x

index = np.random.permutation(node_size)
input_data1 = input_lable[index]
lable_data1 = lable_Y[index]

sen_input_lable = input_data1.reshape((data_len, args.kstep, node_size))

input_data2 = sen_input_lable[0:2000]
lable_data = lable_data1[0:2000]


one_hot_lable = np.zeros((2000, 17))
for i in range(lable_data.shape[0]):
    x = lable_data[i]
    one_hot_lable[i, x.astype(int)] = 1

results = []


for ii in xrange(args.epochs):
    t0 = time.time()
    loss, max_margin_loss = 0., 0.

    print("epoch: ", ii)
    for b in tqdm(xrange(batches_per_epoch)):
        sen_input = sen_gen.next()

        neg_input = neg_gen.next()
        #sen_input2 = sen_gen2.next()

        #sen_input = sen_input2[:,0:-1].reshape((args.batch_size, args.kstep, node_size))
        sen_input = sen_input.reshape((args.batch_size, args.kstep, node_size))
        neg_input = neg_input.reshape((args.batch_size, args.neg_size, args.kstep, node_size))
        # lable_data = sen_input2[:,-1]
        #
        #
        # one_hot_lable = np.zeros((lable_data.shape[0], 17))
        # for i in range(lable_data.shape[0]):
        #     x = lable_data[i]
        #     one_hot_lable[i, x.astype(int)] = 1



        #batch_loss, batch_max_margin_loss = model_auto.train_on_batch([sen_input, neg_input, sen_input2[:,-1]], [np.ones((args.batch_size, 1)), np.ones((args.batch_size, 1))])
        #batch_loss, batch_max_margin_loss = model_auto.train_on_batch([sen_input, one_hot_lable],
        #                                                                 np.ones((args.batch_size, 1)))

        batch_loss, batch_max_margin_loss = model_auto.train_on_batch([sen_input, neg_input], np.ones((args.batch_size, 1)))
        #batch_loss, batch_max_margin_loss = model_lable.train_on_batch(input_data2, one_hot_lable)


        # batch_loss, batch_max_margin_loss = model_auto.train_on_batch([sen_input, neg_input],  [one_hot_lable, np.ones((args.batch_size, 1))])



        loss += batch_loss / batches_per_epoch
        max_margin_loss += batch_max_margin_loss / batches_per_epoch

    # clf_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # node_size = len(vectors)
    # train_x = np.array([vectors[x] for x in X])
    #
    # reshaped_train_x = train_x.reshape((train_x.shape[0], args.kstep, node_size))
    # train_x = low_encoder.predict(reshaped_train_x)
    #
    # for clf_one in clf_list:
    #     print "Training classifier using {:.2f}% nodes...".format(clf_one * 100)
    #     clf = Classifier(vectors=train_x, clf=LogisticRegression())
    #     results.append(clf.split_train_evaluate(X, Y, clf_one)

    tr_time = time.time() - t0


###############################################################################################################################
## classification evaluation
# np_result = np.array(results)
# best_result = np_result.reshape(args.epochs, 9)
# best_result_sum = np.sum(np_result.reshape(args.epochs, 9),1)
# print(best_result)
#
# results_file = open('0127_best_result_wiki.txt', 'w')
# results_file.write(" %s\n" %  best_result)
#
# np.savetxt('best_result_wiki.out', best_result, delimiter='\t')  # X is an array

clf_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
node_size = len(vectors)
train_x = np.array([vectors[x] for x in X])

reshaped_train_x = train_x.reshape((train_x.shape[0], args.kstep, node_size))
train_x = low_encoder.predict(reshaped_train_x)


for clf_one in clf_list:
    print "Training classifier using {:.2f}% nodes...".format(clf_one*100)
    clf = Classifier(vectors=train_x, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, clf_one)


# y_lable = np.array(Y)
# print(train_x.shape)
# print(type(train_x))
# print(type(y_lable))
#
# np.savetxt('train_x.out', train_x, delimiter='\t')   # X is an array
# np.savetxt('train_Y.out', y_lable.astype(int), delimiter='\t')   # X is an array
#
#
# print(results)
# results_file = open('citeseer_result_0.3.txt', 'w')
# for item in results:
#     results_file.write("%s\n" % item)
