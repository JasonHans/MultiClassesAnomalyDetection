def metrics(y_test, y_pred, num_classes):
    M = [[0 for i in range(num_classes)] for j in range(num_classes)]
    for i in range(len(y_test)):
        if y_test[i] == 0:
            if y_pred[i] == 0:
                M[0][0] += 1
            elif y_pred[i] == 1:
                M[0][1] += 1
            elif y_pred[i] == 2:
                M[0][2] += 1
            elif y_pred[i] == 3:
                M[0][3] += 1
            else:
                M[0][4] += 1
        elif y_test[i] == 1:
            if y_pred[i] == 0:
                M[1][0] += 1
            elif y_pred[i] == 1:
                M[1][1] += 1
            elif y_pred[i] == 2:
                M[1][2] += 1
            elif y_pred[i] == 3:
                M[1][3] += 1
            else:
                M[1][4] += 1
        elif y_test[i] == 2:
            if y_pred[i] == 0:
                M[2][0] += 1
            elif y_pred[i] == 1:
                M[2][1] += 1
            elif y_pred[i] == 2:
                M[2][2] += 1
            elif y_pred[i] == 3:
                M[2][3] += 1
            else:
                M[2][4] += 1
        elif y_test[i] == 3:
            if y_pred[i] == 0:
                M[3][0] += 1
            elif y_pred[i] == 1:
                M[3][1] += 1
            elif y_pred[i] == 2:
                M[3][2] += 1
            elif y_pred[i] == 3:
                M[3][3] += 1
            else:
                M[3][4] += 1
        else:
            if y_pred[i] == 0:
                M[4][0] += 1
            elif y_pred[i] == 1:
                M[4][1] += 1
            elif y_pred[i] == 2:
                M[4][2] += 1
            elif y_pred[i] == 3:
                M[4][3] += 1
            else:
                M[4][4] += 1
    print(M)
    n = len(M)
    for i in range(len(M[0])):
        rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
        try:
            print('class %d' % i, 'precision: %s' % (M[i][i]/float(colsum)), 'recall: %s' % (M[i][i]/float(rowsum)),
                  'f1-score: %s' % (2*(M[i][i]/float(colsum))*(M[i][i]/float(rowsum))/(M[i][i]/float(colsum)+M[i][i]/float(rowsum))))
        except ZeroDivisionError:
            print('class %d' % i, 'precision: %s' % 0, 'recall: %s' %0, 'f1-score: %s' %0)