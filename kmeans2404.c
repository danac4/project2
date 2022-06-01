#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>
#include <unistd.h>


#define Mem_Assertion(x) if (!(x)){printf("An Error Has Occurred\n"); abort();}

static PyObject* fit(PyObject *self, PyObject *args);

/* Cluster definition:
count = number of vectors assigned to the cluster 
newPoints = sum of all vectors assigned to the cluster, instead of matrix of all vectors
centroid = current centroid vector
 */
typedef struct{ 
    int count;
    double* newPoints;
    double* centroid;
} Cluster;

/* Point definition:
vector = the point's data vector
*/
typedef struct{
    double* vector;
} Point;

/* Frees points memory, used before returning the output to Python */
void free_data_points(int n, Point* points){
    int i;
    if (points != NULL){
        for (i = 0; i < n; i++){
            free(points[i].vector);
        }
        free(points);
    }
}

/* Frees clusters memory, used before returning the output to Python */
void free_clusters(int k, Cluster* clusters){
    int j;
    for (j = 0; j < k; j++){
        free(clusters[j].newPoints);
        free(clusters[j].centroid);
    }
    free(clusters);
}

/* 
Assigns the current point, given as PyList, into the given vector
input: PyList, empty vector in which the data will be saved and dimension
*/
void pyList_to_array(PyObject *list, double* vector, int dim){
    PyObject *item;
    Py_ssize_t coor;
    for(coor = 0; coor < dim; coor++){
        item = PyList_GetItem(list, coor);
        vector[coor] = PyFloat_AsDouble(item);
    }
}

/* 
Allocates memory for k Clusters, and assigns each Cluster a unique centroid, zero vector and count = 0
input: dimension and PyList of centroids
output: list of k Clusters  
*/
Cluster* createClusters(int dim, PyObject *centroids, int k) {
    Py_ssize_t i;
    Cluster *clusters;
    Cluster clus;
    PyObject *item;
    clusters = (Cluster*) calloc(k, sizeof(Cluster));
    Mem_Assertion(clusters != NULL);
    for (i = 0; i < k; i++) {
        int count = 0;
        double* newPoints = (double*) calloc(dim, sizeof(double));
        double* centroid = (double*) calloc(dim, sizeof(double));
        Mem_Assertion(newPoints != NULL && centroid != NULL);
        item = PyList_GetItem(centroids, i); //extract the point in index i
        pyList_to_array(item, centroid, dim); //assign the point to the new centroid
        clus.count = count;
        clus.newPoints = newPoints;
        clus.centroid = centroid;
        *(clusters + i) = clus;
    }
    return clusters;
}


void free_memory(int k, int n, Point* points, Cluster* clusters){
    free_data_points(n, points);
    free_clusters(k, clusters);
}

/* 
Allocates points memory by dimension and n
input: dimension and size
output: an empty matrix of size n*m
 */
Point* allocate_mem(int dim, int n){
    int i;
    double* v;
    Point* points = (Point*)calloc(n, sizeof(Point));
    Mem_Assertion(points != NULL);
    for(i = 0; i < n; i++){
        v = (double*) calloc(dim,sizeof(double));
        Mem_Assertion(v != NULL);
        points[i].vector = v;
    }
    return points;
}

/* Assigns the n data points, stored in points_list, to the Points in points */
void create_matrix(PyObject* points_list, Point* points, int dim, int n) {
    Py_ssize_t i;
    PyObject *item;
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(points_list, i);
        pyList_to_array(item, points[i].vector, dim);
    }
}

/*
 Computes the distance between the given point and the given centroid
 input: point, centroid, dimension
 output: distance, double  
 */
double distance(Point* point1, const double* centroid, int dim){
    int i;
    double sum, p;
    sum = 0;
    for (i = 0; i < dim; i++){
        p = point1->vector[i] - *(centroid + i);
        sum += p*p;
    }
    return sqrt(sum);
}

/*
 Finds the closest cluster to the given point
 input: point, list of clusters, dimension, k
 output: index of the required cluster, int  
*/
int min_distance(Point* point, Cluster* clusters, int dim, int k){
    int min_index, i;
    double min_val, curr;
    min_index = 0;
    min_val = distance(point, clusters[0].centroid, dim); //define the distance from the first cluster as min_val
    for (i = 1; i < k; i++){
        curr = distance(point, clusters[i].centroid, dim);
        if (curr < min_val){
            min_val = curr;
            min_index = i;
        }
    }
    return min_index;
}

/* Compute the euclidean norm */
double euclidean_norm(const double* vector, int dim){
    int i;
    double result,p;
    result = 0;
    for (i = 0; i < dim; i++){
        p = vector[i];
        result += p*p;
    }
    return sqrt(result);
}

/*
 Add the given point to the given cluster 
 The insert operation is implemented by adding each coordinate to its adequate index in new_points vector
 input: point, cluster, dimension
*/
void add_point(Point* point, Cluster* cluster, int dim){
    int i;
    for (i = 0; i < dim; i++) {
        cluster->newPoints[i] += point->vector[i];
    }
    cluster->count += 1; //increase the number of points in the given cluster by 1 
}

/*
 Update the centroid of the given cluster by computing the average value of each coordinate in new_points
 Check convergence
 input: cluster, dimension, an empty vector used for holding the result of the computation, eps
 output: indicator for convergence, int
 */
int centroid_update(Cluster* cluster, int dim, double *tmp_vector, double eps){
    int has_changed, i, l;
    double norm_check;
    has_changed = 1;
    if (cluster->count == 0){
        return 1;
    }
    for (i = 0; i < dim; i++) {
        tmp_vector[i] = cluster->newPoints[i]/cluster->count;
    }
    norm_check = euclidean_norm(cluster->centroid, dim) - euclidean_norm(tmp_vector, dim);
    if (norm_check >= eps || norm_check <= -eps){
        has_changed = 0;
    }
    for (l = 0; l < dim; l++) {
        cluster->centroid[l] = tmp_vector[l];
        cluster->newPoints[l] = 0;
    }
    cluster-> count = 0;
    return has_changed;
}

/*
 Update the centroids of the clusters, and initialize count and new_points for each cluster
 Called at the end of each iteration in Kmeans
 input: list of clusters, k, dimension, epsilon
 output: indicator for convergence, int
 */
int clusters_update(Cluster* clusters, int k, int dim, double eps) {
    int changed, i, epsilon_indicator;
    double *tmp_vector;
    changed = 1;
    tmp_vector = (double *) calloc(dim, sizeof(double)); //Create a temporary vector to hold the new values before assigning them to the centroid
    Mem_Assertion(tmp_vector != NULL);
    for (i = 0; i < k; i++) {
        epsilon_indicator = centroid_update(&clusters[i], dim, tmp_vector, eps);
        changed = ((changed) && (epsilon_indicator));
    }
    free(tmp_vector); //free 'tmp_vector'
    return changed;
}

/*
 array_to_Pylist transformes kmeans final centroids array of arrays to a pylists of pylists containing Python floats
 input: clusters = list of clusters, k, dim = dimension
 output: PyList = a python lists of lists containing final centroids points as PyFloat
*/
PyObject* array_to_PyList(Cluster *clusters, int k, int dim){
    Py_ssize_t i, j;
    PyObject *PyList = PyList_New((Py_ssize_t)(k));
    PyObject *item;
    for(i = 0; i < k; i++){
        item = PyList_New((Py_ssize_t)(dim));
        for(j = 0; j < dim; j++){
            PyList_SetItem(item, j, PyFloat_FromDouble(clusters[i].centroid[j]));
        }
        PyList_SetItem(PyList, i, item);
    }
    return PyList;
}

/*
 kmeans algorithem implementaion
 input: max_iter, n, eps, clusters = list of clusters, points = list of data points, dim = dimension, k
*/
void kmeans(int max_iter, int n, double eps, Cluster* clusters, Point* points, int dim, int k) {
    int epsilon_check, iter, i, index;
    epsilon_check = 0; // epsilon_check is an identicator that has a value of 1 iff the euclidean norm of each centroids doesn't change by more then epsilon.
    iter = 0; //iterations counter
    while ((iter < max_iter) && (1 - epsilon_check)) {
        for (i = 0; i < n; i++) {
            index = min_distance(&points[i], clusters, dim, k);
            add_point(&points[i], &clusters[index], dim);
        }
        epsilon_check = clusters_update(clusters, k, dim, eps);
        iter++;
    }
}

/* fit function:
input: k, n - the number of points in data, dim - the dimension of each data point, max_iter, eps - provided epsilon, 
centroids_list - python list of intial centroids, points_list - a python list of lists where each list is a data point in the given data
output: raises an error if one or more of the input arguments is not in correct format,
otherwise return final- a python list of lists with final centroids caculated by k-means algorithem implemented in kmeans.
 */
static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject *centroids_list, *points_list, *final;
    double eps;
    int k, n, dim, max_iter;
    Point *points;
    Cluster *clusters;
    if (!PyArg_ParseTuple(args, "iiiidOO", &k, &n, &dim, &max_iter, &eps, &centroids_list, &points_list)) {
        return NULL;
    }
    points = allocate_mem(dim, n);
    create_matrix(points_list, points, dim, n);
    clusters = createClusters(dim, centroids_list, k);
    kmeans(max_iter, n, eps, clusters, points, dim, k);
    final = array_to_PyList(clusters, k, dim);
    free_memory(k, n, points, clusters);
    return final;
}

int main(int argc, char *argv[]) {
    return 1;
}

/* API */

#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }

static PyMethodDef _methods[] = {
        FUNC(METH_VARARGS, fit, "kmeans implementation"),
        {NULL, NULL, 0, NULL}   /* sentinel */
};


static struct PyModuleDef _moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        NULL,
        -1,
        _methods
};


PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&_moduledef);
}
