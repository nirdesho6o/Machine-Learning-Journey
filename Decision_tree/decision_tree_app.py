import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.tree import DecisionTreeClassifier, plot_tree # Imported plot_tree
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y
    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Decision Tree Classifier")

dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Binary', 'Multiclass')
)

# --- Decision Tree Hyperparameters ---
criterion = st.sidebar.selectbox(
    'Criterion',
    ('gini', 'entropy', 'log_loss')
)

splitter = st.sidebar.selectbox(
    'Splitter',
    ('best', 'random')
)

max_depth = int(st.sidebar.number_input('Max Depth (0 for None)', value=0, min_value=0, step=1))

min_samples_split = int(st.sidebar.number_input('Min Samples Split', value=2, min_value=2, step=1))

min_samples_leaf = int(st.sidebar.number_input('Min Samples Leaf', value=1, min_value=1, step=1))

max_features = st.sidebar.selectbox(
    'Max Features',
    ('None', 'sqrt', 'log2') 
)

max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes (0 for None)', value=0, min_value=0, step=1))

min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease', value=0.0, min_value=0.0, step=0.01)
# -------------------------------------

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    # Pre-processing params for "None" values
    p_max_depth = None if max_depth == 0 else max_depth
    p_max_features = None if max_features == 'None' else max_features
    p_max_leaf_nodes = None if max_leaf_nodes == 0 else max_leaf_nodes

    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=p_max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=p_max_features,
        max_leaf_nodes=p_max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        random_state=42
    )
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # 1. Plot Decision Boundary
    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)
    
    st.subheader("Accuracy for Decision Tree: " + str(round(accuracy_score(y_test, y_pred), 2)))

    # 2. Plot Tree Structure
    st.subheader("Tree Structure Visualization")
    # We create a new figure for the tree plot
    fig_tree, ax_tree = plt.subplots(figsize=(20, 10)) 
    
    # plot_tree returns the annotations, we suppress output with _
    _ = plot_tree(
        clf, 
        filled=True, 
        ax=ax_tree, 
        feature_names=['Col1', 'Col2'],
        class_names=[str(c) for c in clf.classes_],
        rounded=True,
        fontsize=10 
    )
    
    st.pyplot(fig_tree)