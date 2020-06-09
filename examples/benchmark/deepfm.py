# import comet_ml in the top of your file
from comet_ml import Experiment
import tensorflow as tf   
# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="ej6XeyCVjqHM8uLDNj5VGrzjP",
                        project_name="benchmark", workspace="pingsutw")
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import time
from deepctr.models import DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":
    t0 = time.time()
    names = "label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26".split(",")
    feats = ["C"+str(i) for i in range(1, 27)] + ["I"+str(i) for i in range(1, 14)]
    print("run deepfm 1M : ")
    data = pd.read_csv('/root/ctr/benchmark_data/1M.txt', header=None, names=names,sep='\t')
    experiment.add_tag("1M")
    experiment.add_tag("deepfm")
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4)
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_hidden_units=(400,400,400),dnn_dropout=0.5)
    model.compile("adam", "binary_crossentropy",
                metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    experiment.log_metric("LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    experiment.log_metric("ROC", round(roc_auc_score(test[target].values, pred_ans), 4))
    t1 = time.time()
    print("run time", t1-t0)
    experiment.log_metric("Time", t1-t0)
