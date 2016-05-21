# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score

# 这里的Kaggle自行车租赁预测比赛也同样是一个很有趣的问题，之所以要把它单拎出来，讲一讲，2个原因和泰坦尼克号之灾是一样的，另外一个原因是，这是一个连续值预测的问题，本着各类问题我们都要覆盖一下的标准，咱们一起来看看这个问题。
# 这是一个城市自行车租赁系统，提供的数据为2年内华盛顿按小时记录的自行车租赁数据，其中训练集由每个月的前19天组成，测试集由20号之后的时间组成（需要我们自己去预测）。
def set_ch():
    mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def load_data(filename):
    df_train = pd.read_csv(filename, header=0)
    df_train.head(10)
    # print df_train.dtypes
    #让它告诉我们形状
    df_train.shape
    # 我们总结一下，总共有10886行，同时每一行有12列不同的信息
    # 那个，记得我们说过的脏数据问题吧，所以呢，我们看看有没有缺省的字段
    df_train.count()
    # →_→可见万恶的资本主义郭嘉的记录系统多么严谨完善，居然就没有缺省值
    type(df_train.datetime)
    # 咱们第一个来处理时间，因为它包含的信息总是非常多的，毕竟变化都是随着时间发生的嘛
    # 把月、日、和 小时单独拎出来，放到3列中
    df_train['month'] = pd.DatetimeIndex(df_train.datetime).month
    df_train['day'] = pd.DatetimeIndex(df_train.datetime).dayofweek
    df_train['hour'] = pd.DatetimeIndex(df_train.datetime).hour
    # 再看
    df_train.head(10)
    # 那个，既然时间大串已经被我们处理过了，那这个字段放着太占地方，干脆就不要了吧
    # 先上一个粗暴的版本，咱们把注册租户和未注册租户也先丢掉，回头咱们再看另外一种处理方式
    # 那个，保险起见，咱们还是先存一下吧
    df_train_origin = df_train
    # 抛掉不要的字段
    df_train = df_train.drop(['datetime', 'casual', 'registered'], axis=1)
    # 看一眼
    df_train.head(5)
    # print df_train.shape
    # (10886, 12)
    return df_train, df_train_origin

def splitdata(df_train_data):
    cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=3, test_size=0.2, random_state=0)
    return cv

def use_algorithm(df_train_data, df_train_target, cv):
    # 各种模型来一圈

    # print "岭回归"
    # for train, test in cv:
    #     svc = linear_model.Ridge().fit(df_train_data[train], df_train_target[train])
    #     print("train score: {0:.3f}, test score: {1:.3f}\n".format(
    #         svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))

    # print "支持向量回归/SVR(kernel='rbf',C=10,gamma=.001)"
    # for train, test in cv:
    #     svc = svm.SVR(kernel='rbf', C=10, gamma=.001).fit(df_train_data[train], df_train_target[train])
    #     print("train score: {0:.3f}, test score: {1:.3f}\n".format(
    #         svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))

    print "随机森林回归/Random Forest(n_estimators = 100)"
    for train, test in cv:
        svc = RandomForestRegressor(n_estimators=100).fit(df_train_data[train], df_train_target[train])
        print("train score: {0:.3f}, test score: {1:.3f}\n".format(
            svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))

# 岭回归
# train score: 0.339, test score: 0.332
#
# train score: 0.330, test score: 0.370
#
# train score: 0.342, test score: 0.320
#
# 支持向量回归/SVR(kernel='rbf',C=10,gamma=.001)
# train score: 0.417, test score: 0.408
#
# train score: 0.406, test score: 0.452
#
# train score: 0.419, test score: 0.390
#
# 随机森林回归/Random Forest(n_estimators = 100)
# train score: 0.981, test score: 0.866
#
# train score: 0.981, test score: 0.880
#
# train score: 0.981, test score: 0.870
def getScore(df_train_data, df_train_target):
    X = df_train_data
    y = df_train_target
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    tuned_parameters = [{'n_estimators': [10, 100, 500]}]
    scores = ['r2']
    for score in scores:
        print score
        clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)

        print("别！喝！咖！啡！了！最佳参数找到了亲！！：")
        print ""
        # best_estimator_ returns the best estimator chosen by the search
        print(clf.best_estimator_)
        print ""
        print("得分分别是:")
        print ""
        # grid_scores_的返回值:
        #    * a dict of parameter settings
        #    * the mean score over the cross-validation folds
        #    * the list of scores for each fold
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print ""
        return X, y
# r2
# 别！喝！咖！啡！了！最佳参数找到了亲！！：
#
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#                       max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
#                       verbose=0, warm_start=False)
#
# 得分分别是:
#
# 0.848 (+/-0.007) for {'n_estimators': 10}
#     0.863 (+/-0.005) for {'n_estimators': 100}
#     0.863 (+/-0.006) for {'n_estimators': 500}

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def useDataAnalysis(df_train_origin):
    # 风速
    df_train_origin.groupby('windspeed').mean().plot(y='count', marker='o')
    # plt.show()
    # 湿度
    df_train_origin.groupby('humidity').mean().plot(y='count', marker='o')
    # plt.show()
    # 温度
    df_train_origin.groupby('temp').mean().plot(y='count', marker='o')
    # plt.show()
    # 温度湿度变化
    df_train_origin.plot(x='temp', y='humidity', kind='scatter')
    # plt.show()
    # scatter一下各个维度
    fig, axs = plt.subplots(2, 3, sharey=True)
    df_train_origin.plot(kind='scatter', x='temp', y='count', ax=axs[0, 0], figsize=(16, 8), color='magenta')
    df_train_origin.plot(kind='scatter', x='atemp', y='count', ax=axs[0, 1], color='cyan')
    df_train_origin.plot(kind='scatter', x='humidity', y='count', ax=axs[0, 2], color='red')
    df_train_origin.plot(kind='scatter', x='windspeed', y='count', ax=axs[1, 0], color='yellow')
    df_train_origin.plot(kind='scatter', x='month', y='count', ax=axs[1, 1], color='blue')
    df_train_origin.plot(kind='scatter', x='hour', y='count', ax=axs[1, 2], color='green')

    sns.pairplot(df_train_origin[["temp", "month", "humidity", "count"]], hue="count")

    # 来看看相关度咯
    corr = df_train_origin[['temp', 'weather', 'windspeed', 'day', 'month', 'hour', 'count']].corr()
    print corr

    # 用颜色深浅来表示相关度
    plt.figure()
    plt.matshow(corr)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    set_ch()
    df_train, df_train_origin = load_data('kaggle_bike_competition_train.csv')
    # 分成2部分:
    # 1. df_train_target：目标，也就是count字段。
    # 2. df_train_data：用于产出特征的数据
    df_train_target = df_train['count'].values
    df_train_data = df_train.drop(['count'], axis=1).values
    # print 'df_train_data shape is ', df_train_data.shape # df_train_data shape is  (10886L, 11L)
    # print 'df_train_target shape is ', df_train_target.shape # df_train_target shape is  (10886L,)
# df_train_data shape is  (10886L, 11L)
# df_train_target shape is  (10886L,)
# 下面的过程会让你看到，其实应用机器学习算法的过程，多半是在调参，各种不同的参数会带来不同的结果（比如正则化系数，比如决策树类的算法的树深和棵树，比如距离判定准则等等等等）
    # 总得切分一下数据咯（训练集和测试集）
    # cross_valid = splitdata(df_train_data)
# 数据量不算大，世界那么大，你想去看看，没钱看不成；模型这么多，你尽量试试总可以吧。
# 咱们依旧会使用交叉验证的方式（交叉验证集约占全部数据的20%）来看看模型的效果，我们会试 支持向量回归/Suport Vector Regression, 岭回归/Ridge Regression 和 随机森林回归/Random Forest Regressor。每个模型会跑3趟看平均的结果。
# 什么，你说这些模型还没讲，你都不懂？没关系，先练练手试试咯，学会读文档嘛。
# 支持向量回归  http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
# 岭回归  http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
# 随机森林回归  http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#     use_algorithm(df_train_data, df_train_target, cross_valid)
# 不出意料，随机森林回归获得了最佳结果。
# 不过，那个，大家有没有想过，也有可能是你的参数设置的不对啊？这个，留给大家自己去试试咯，试完告诉我，哈哈
# 放个大招，好多同学问参数咋啊？我们有一个工具可以帮忙，叫做GridSearch，可以在你喝咖啡的时候，帮你搬搬砖，找找参数
#     X, y = getScore(df_train_data, df_train_target)
# 你看到咯，Grid Search帮你挑参数还是蛮方便的，你也可以大胆放心地在刚才其他的模型上试一把。
# 而且要看看模型状态是不是，过拟合or欠拟合
# 依旧是学习曲线
#     title = "Learning Curves (Random Forest, n_estimators = 100)"
#     cv = cross_validation.ShuffleSplit(df_train_data.shape[0], n_iter=10, test_size=0.2, random_state=0)
#     estimator = RandomForestRegressor(n_estimators=100)
    # plt = plot_learning_curve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=4)
    # plt.show()
# 看出来了吧，训练集和测试集直接间隔那么大，这。。。一定是过拟合了
# 随机森林这种算法学习能力非常强啦，大家从最上面对比各个模型得分的时候也可以看到，训练集和测试集的得分也是差蛮多的，过拟合还蛮明显。所以，我能说什么呢，你用了核弹去消灭蝗虫，然后土壤也有点长不出植物了
# so, 过拟合咱们怎么办来着？你来回答下？忘了？那还不去翻翻ppt
# 尝试一下缓解过拟合，当然，未必成功
#     print "随机森林回归/Random Forest(n_estimators=200, max_features=0.6, max_depth=15)"
#     for train, test in cv:
#         svc = RandomForestRegressor(n_estimators=200, max_features=0.6, max_depth=15).fit(df_train_data[train], df_train_target[train])
#         print("train score: {0:.3f}, test score: {1:.3f}\n".format(
#             svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))
# 不知道大家有没有印象，我们之前说了，我们把“注册用户”和“未注册用户”加一块儿做的预测。
# 另外一个思路是，咱们可以试着分开这两部分，分别预测一下，再求和嘛。
# 话说，特征和对应的“注册”和“未注册”用户都有了，这个部分就当做作业吧，大家试试。
# 看你们自己的咯
    df_train_registered = df_train_origin.drop(['datetime', 'casual', 'count'], axis=1)
    df_train_casual = df_train_origin.drop(['datetime', 'count', 'registered'], axis=1)
    df_train_registered.head()
    # 听说有同学问，为啥这个例子中没有数据分析，咳咳，那好吧，补充一下。那个，分析得到的结果，你们观察观察，看看有什么角度可以帮忙改善一下特征或者模型，看好你们^_^¶
    useDataAnalysis(df_train_origin)