from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np


''' 엘보우 방법으로 최적 클러스터 개수 찾기 '''
def find_optimal_k_elbow(vectors, k_range=(2, 15), plot=True):
    inertias = []
    k_list = list(range(k_range[0], k_range[1] + 1))

    # k별 inertia 계산 (클러스터 내 거리 제곱합)
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(vectors)
        inertias.append(km.inertia_)

    # 그래프 시각화
    if plot:
        plot_elbow(k_list, inertias)

    # KneeLocator로 꺾이는 지점 찾기
    kn = KneeLocator(k_list, inertias, curve='convex', direction='decreasing')
    return kn.knee if kn.knee is not None else k_list[0]

''' Elbow 방식 시각화 '''
def plot_elbow(k_list, inertias):
    plt.figure(figsize=(8, 4))
    plt.plot(k_list, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

''' 실루엣 점수 기반으로 최적 클러스터 개수(k) 찾기 '''
def find_optimal_k_silhouette(vectors, k_range=(2, 15), plot=True):
    best_score = -1
    best_k = k_range[0]
    best_labels = None

    for k in range(k_range[0], k_range[1] + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(vectors)
            score = silhouette_score(vectors, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except:
            continue  # silhouette 계산 실패한 경우 무시

    if plot and best_labels is not None:
        plot_silhouette(vectors, best_k, best_labels)

    return best_k

''' 실루엣 점수 시각화 '''
def plot_silhouette(X, n_clusters, cluster_labels):
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for spacing between clusters

    ax.set_title(f"Silhouette plot for {n_clusters} clusters (avg={silhouette_avg:.3f})")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    plt.show()

''' 엘보우와 실루엣 점수로 최적 k 결정 '''
def find_optimal_k_combined(vectors, plot, k_range=(2, 15), strategy="silhouette"):
    elbow_k = find_optimal_k_elbow(vectors, k_range, plot)
    silhouette_k = find_optimal_k_silhouette(vectors, k_range, plot)
    avg_k = round((elbow_k + silhouette_k) / 2)
    
    print(f"Elbow method optimal k: {elbow_k}")
    print(f"Silhouette method optimal k: {silhouette_k}")
    print(f"Average k: {avg_k}")

    if elbow_k == silhouette_k:
        return elbow_k

    if strategy == "average":
        return avg_k
    elif strategy == "elbow":
        return elbow_k
    else:  # default: silhouette
        return silhouette_k
