{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "1647f706-4d9f-48d7-bc04-9885a282aacf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Transformed Data:\n",
      " [[ 2.54712114 -4.37132521]\n",
      " [ 1.41835139 -7.79844924]\n",
      " [ 2.06665508 -4.6196175 ]\n",
      " [ 0.06087201 -7.07805636]\n",
      " [ 1.33443258 -9.26371897]\n",
      " [ 0.62525689 -5.36449435]\n",
      " [ 2.85974959 -7.05357239]\n",
      " [ 4.21722897 -7.77396526]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "class ManualLDA:\n",
    "    def __init__(self, n_components=None):\n",
    "        self.n_components = n_components\n",
    "\n",
    "    def fit(self, X, y):\n",
    "        n_samples, n_features = X.shape\n",
    "        \n",
    "\n",
    "        class_labels = np.unique(y)\n",
    "        n_classes = len(class_labels)\n",
    "        \n",
    "     \n",
    "        mean_overall = np.mean(X, axis=0)\n",
    "        \n",
    "    \n",
    "        Sw = np.zeros((n_features, n_features))\n",
    "        Sb = np.zeros((n_features, n_features))\n",
    "        \n",
    "        for c in class_labels:\n",
    "            # Get the samples for this class\n",
    "            X_c = X[y == c]\n",
    "            mean_c = np.mean(X_c, axis=0)\n",
    "            \n",
    "            Sw += np.dot((X_c - mean_c).T, (X_c - mean_c))\n",
    "            \n",
    "           \n",
    "            n_c = X_c.shape[0]  # Number of samples in class c\n",
    "            mean_diff = (mean_c - mean_overall).reshape(-1, 1)\n",
    "            Sb += n_c * np.dot(mean_diff, mean_diff.T)\n",
    "        \n",
    "    \n",
    "        Sw_inv = np.linalg.inv(Sw)\n",
    "        eigvals, eigvecs = np.linalg.eig(np.dot(Sw_inv, Sb))\n",
    "        \n",
    "        \n",
    "        sorted_indices = np.argsort(eigvals)[::-1]\n",
    "        eigvals_sorted = eigvals[sorted_indices]\n",
    "        eigvecs_sorted = eigvecs[:, sorted_indices]\n",
    "        \n",
    "        \n",
    "        if self.n_components is not None:\n",
    "            eigvecs_sorted = eigvecs_sorted[:, :self.n_components]\n",
    "        \n",
    "        self.W = eigvecs_sorted\n",
    "    \n",
    "    def transform(self, X):\n",
    "        return np.dot(X, self.W)\n",
    "    \n",
    "    def fit_transform(self, X, y):\n",
    "        self.fit(X, y)\n",
    "        return self.transform(X)\n",
    "\n",
    "# Example usage:\n",
    "if __name__ == \"__main__\":\n",
    "   \n",
    "    X = np.array([[4, 2], [6, 8], [4, 3], [5, 9], [7, 10], [4, 6], [6, 5], [7, 4]])\n",
    "    y = np.array([0, 0, 0, 1, 1, 1, 2, 2])\n",
    "\n",
    "\n",
    "    lda = ManualLDA(n_components=2)\n",
    "    X_lda = lda.fit_transform(X, y)\n",
    "    \n",
    "    print(\"Transformed Data:\\n\", X_lda)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3369178-f48d-41c1-919d-87241c907e33",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
