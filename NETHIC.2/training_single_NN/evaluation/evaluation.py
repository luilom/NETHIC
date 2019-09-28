
import pandas as pd
import statistics
import plotly.graph_objects as go
import plotly.offline as offline


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

kinds = ['doc2vec-BOW','BOW','doc2vec']
for kind in kinds:
    results = pd.read_pickle("../results_single_NN_"+str(kind)+"/results_all_categories.pkl")

    avg_accuracy_training_test_k_fold_cv = list()
    training_accuracies = list()
    test_accuracies = list()
    f1_score = list()
    precision = list()
    recall = list()
    table_content = list()
    for key in results.keys():
        avg_accuracy_training_test_k_fold_cv.append(truncate(results[key]["avg_accuracy_training_test_k_fold_cv"],2))
        training_accuracies.append(truncate(results[key]["training_accuracies"],2))
        test_accuracies.append(truncate(results[key]["test_accuracies"],2))
        f1_score.append(truncate(statistics.mean(results[key]["f1_score"]),2))
        precision.append(truncate(statistics.mean(results[key]["precision"]),2))
        recall.append(truncate(statistics.mean(results[key]["recall"]),2))


    table_content.append(list(map(lambda x : x.replace("_"," "), results.keys())))
    table_content.append(avg_accuracy_training_test_k_fold_cv)
    table_content.append(training_accuracies)
    table_content.append(test_accuracies)
    table_content.append(f1_score)
    table_content.append(precision)
    table_content.append(recall)


    fig = go.Figure(data=[go.Table(header=dict(
        values=['Category','Cross Validation', 'Training Accuracy', 'Test Accuracy', 'F1-Score', 'Precision', 'Recall'],
        align='left'),cells=dict(values=table_content,align='left'))])
    fig.update_layout(width=1000, height=1000)
    fig.show()


    plotly.offline.init_notebook_mode()

    offline.iplot(data=[go.Table(header=dict(
        values=['Category','Cross Validation', 'Training Accuracy', 'Test Accuracy', 'F1-Score', 'Precision', 'Recall'], align='left'),
        cells=dict(values=table_content,align='left'),image='svg')])


