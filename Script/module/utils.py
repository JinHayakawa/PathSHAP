import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(y_test, y_pred, classes, filename, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(25, 20)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm=confusion_matrix(y_test, y_pred, np.unique(y_test))
    plt.figure(figsize=figsize, facecolor="white")

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.subplots_adjust(bottom=0.1, wspace=0.05)
    plt.savefig(filename+".png")
    plt.savefig(filename+".eps")


###################################
# Data                            #
###################################
def get_class_distribution(obj):
    """Count numbers of each class to class weighting"""
    count_dict = {x:int(0) for x in obj}
    
    for x in obj:
        count_dict[x] += 1
            
    return count_dict

def create_gsea_dataset(df, label_idx, gene_id_type="symbol", filename="gctfile", case_filename="gctcase"):
    """
    gene_id: str:
        symbol or entrezgene
    """
    # 全Index
    merged_idx = list()
    [merged_idx.extend(label_idx[k]) for k in label_idx]

    # Dictionary to labe cases
    reverse_dict = {}
    i = 0
    for k,v in label_idx.items():
        for idx in v:
            reverse_dict[idx] = i
        i+=1
        
    # data extraction
    gct_df = df.loc[:, merged_idx].copy()
    #gct_df.index = gct_df.index.map(gene_convert_dict)
    N_gene   = gct_df.shape[0]
    N_sample = gct_df.shape[1]
    N_class  = len(label_idx) 
    labels = [k for k in label_idx]
    labels_txt   = " ".join(labels)
    classes  = gct_df.columns.map(reverse_dict)
    classes_txt  = " ".join([str(c) for c in classes])
    
    # for gct file
    gct_df.index.name = "NAME"
    gct_df.insert(0, "DESCRIPTION", "na")
    gct_txt = gct_df.to_csv(sep='\t')
    gct_txt = f"#1.2\n{N_gene}\t{N_sample}\n{gct_txt}"
    
    os.makedirs(filename, exist_ok=True)
    
    with open(os.path.join(filename, filename+".gct"), "w") as f:
        f.write(gct_txt)
    
    # for cls file
    cls_txt = f"{N_sample} {N_class} 1\n# {labels_txt}\n{classes_txt}"

    with open(os.path.join(filename, filename+".cls"), "w") as f:
        f.write(cls_txt)
 
    print("Finish writing GCT file and Case file.")
    return gct_df.drop("DESCRIPTION", axis=1), list(map(lambda x: labels[x], classes))