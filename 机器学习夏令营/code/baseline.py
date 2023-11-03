# %%
import sweetviz as sv
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import scikitplot as skplt
import warnings

warnings.filterwarnings('ignore')


def readData(path='用户新增预测挑战赛公开数据/'):
    train_data = pd.read_csv(path + 'train.csv')
    test_data = pd.read_csv(path + 'test.csv')
    return train_data, test_data


def eda(df, name, feature=None, target=None):
    feature_config = sv.FeatureConfig(force_text=feature, force_cat=feature)
    sweet_report = sv.analyze(df, feat_cfg=feature_config, target_feat=target)
    sweet_report.show_html(f'{name}.html')


def FE(train_data, test_data):
    eidmean = train_data.groupby('eid')['target'].mean()

    def udmap_onethot(d):
        v = np.zeros(9)
        if d == 'unknown':
            return v

        d = eval(d)
        for i in range(1, 10):
            if 'key' + str(i) in d:
                v[i - 1] = d['key' + str(i)]

        return v

    def makeFeatures(df):
        df['common_ts'] = pd.to_datetime(df['common_ts'], unit='ms')
        udmapDf = pd.DataFrame(np.vstack(df['udmap'].apply(udmap_onethot)))
        udmapDf.columns = ['key' + str(i) for i in range(1, 10)]
        df = pd.concat([df, udmapDf], axis=1)
        df['udmap_isunknown'] = (df['udmap'] == 'unknown').astype(int)
        df['eid_freq'] = df['eid'].map(df['eid'].value_counts())
        df['eid_mean'] = df['eid'].map(eidmean)
        df['common_ts_hour'] = df['common_ts'].dt.hour
        df = df.drop(['udmap', 'common_ts'], axis=1)
        return df

    train_data = makeFeatures(train_data)
    test_data = makeFeatures(test_data)
    return train_data, test_data


def Submit(clf, test_data, name):
    pd.DataFrame({
        'uuid': test_data['uuid'],
        'target': clf.predict(test_data.drop(['uuid'], axis=1))
    }).to_csv(name + '_submit.csv', index=None)


def lgbCVBag(train_data, test_data):
    def lgb_important(bst):
        fig, ax = plt.subplots(figsize=(15, 10))
        lgb.plot_importance(bst, max_num_features=20, ax=ax, importance_type='gain')
        plt.yticks(fontsize=8)
        plt.xlabel('Feature importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.show()

    def lgbCV(df):
        df = df.drop(['uuid'], axis=1)

        def f1_loss(preds, train_data):
            y_label = train_data.get_label()
            y_pred = [int(i > 0.5) for i in preds]
            f1_score = f1(y_pred, y_label, average='macro')
            return 'f1score', f1_score, True

        X = df[[i for i in df.columns if i not in ['target']]]
        y = df.target
        train_data = lgb.Dataset(X, label=y)

        param = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_leaves': 63,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'learning_rate': 0.1,
            'seed': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

        cvRst = lgb.cv(param, train_data, 1000, nfold=3, stratified=False, return_cvbooster=True, feval=f1_loss)

        # pd.DataFrame(cvRst['f1score-mean']).plot()
        # plt.show()
        #
        bsts = cvRst['cvbooster']
        return bsts

    def Predict(bsts, test_data):
        pred = pd.DataFrame(bsts.predict(test_data.drop(['uuid'], axis=1), num_iteration=bsts.best_iteration)).mean()
        target = [int(i > 0.5) for i in pred]
        submitDf = pd.DataFrame({
            'uuid': test_data['uuid'],
            'target': target
        })
        submitDf.to_csv('submit.csv', index=None)



    bsts = lgbCV(train_data)
    lgb_important(bsts.boosters[0])
    lgb_important(bsts.boosters[1])
    lgb_important(bsts.boosters[2])

    Predict(bsts, test_data)


def RandomForestBag(train_data, test_data):
    def learn_curve(start, end, step, X, y):
        df = pd.DataFrame()
        for i in tqdm(np.arange(start, end, step)):
            rfc = RandomForestClassifier(n_estimators=i, n_jobs=-1, random_state=90)
            score = cross_val_score(rfc, X, y, cv=3, scoring='f1_macro').mean()
            df = pd.DataFrame.append(df, {'score': score, 'i': i}, ignore_index=True)

        df.plot(x='i', y='score')
        print(df.sort_values('score', ascending=False))
        plt.show()
        return df

    train_data = train_data.drop(['uuid'], axis=1)
    X = train_data[[i for i in train_data.columns if i not in ['target']]]
    y = train_data.target
    learn_curve(300, 1000, 100, X, y)
    rfc = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=90)
    rfc.fit(X, y)
    skplt.estimators.plot_feature_importances(rfc, feature_names=X.columns)
    plt.show()
    Submit(rfc, test_data, 'rf2')


# %%
if __name__ == '__main__':
    train_data, test_data = readData()
    # eda(train_data, "user", target='target')
    train_data, test_data = FE(train_data, test_data)

    # Train using LightGBM and produce important features and predictions
    lgbCVBag(train_data, test_data)

    # Train using RandomForest and produce important features and predictions
    RandomForestBag(train_data, test_data)
    

