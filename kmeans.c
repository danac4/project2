#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#define Mem_Assertion(x) if (!(x)){printf("An Error Has Occurred\n"); abort();}

static PyObject* fit(PyObject *self, PyObject *args);

typedef struct{
    int count;
    double* newPoints;
    double* centroid;
} Cluster;

typedef struct{
    double* vector;
} Point;


void free_data_points(int n, Point* points){
    int i;
    if (points != NULL){
        for (i = 0; i < n; i++){
            free(points[i].vector);
        }
        free(points);
    }
}
void free_clusters(int k, Cluster* clusters){
    int j;
    for (j = 0; j < k; j++){
        free(clusters[j].newPoints);
        free(clusters[j].centroid);
    }
    free(clusters);
}


void pyList_to_array(PyObject *list, double* vector, int dim){
    PyObject *item;
    Py_ssize_t coor;
    for(coor = 0; coor < dim; coor++){
        item = PyList_GetItem(list, coor);
        vector[coor] = PyFloat_AsDouble(item);
    }
}


Cluster* createClusters(int dim, PyObject *centroids, int k) {
    int i;
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
        item = PyList_GetItem(centroids, i);
        pyList_to_array(item, centroid, dim);
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

void create_matrix(PyObject* points_list, Point* points, int dim, int n) {
    int i;
    PyObject *item, *coordinate;
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(points_list, i);
        pyList_to_array(item, points[i].vector, dim);
//        for(j = 0; j < dim; j++){
//            coordinate = PyList_GetItem(item, j);
//            points[i].vector[j] = PyFloat_AsDouble(coordinate);
    }
}


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

int min_distance(Point* point, Cluster* clusters, int dim, int k){
    int min_index, i;
    double min_val, curr;
    min_index = 0;
    min_val = distance(point, clusters[0].centroid, dim);
    for (i = 1; i < k; i++){
        curr = distance(point, clusters[i].centroid, dim);
        if (curr < min_val){
            min_val = curr;
            min_index = i;
        }
    }
    return min_index;
}

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


void add_point(Point* point, Cluster* cluster, int dim){
    int i;
    for (i = 0; i < dim; i++) {
        cluster->newPoints[i] += point->vector[i];
    }
    cluster->count += 1;
}


int centroid_update(Cluster* cluster, int dim, double *tmp_vector, int eps){
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


int clusters_update(Cluster* clusters, int k, int dim, int eps) {
    int changed, i, epsilon_indicator;
    double *tmp_vector;
    changed = 1;
    tmp_vector = (double *) calloc(dim, sizeof(double));
    Mem_Assertion(tmp_vector != NULL);
    for (i = 0; i < k; i++) {
        epsilon_indicator = centroid_update(&clusters[i], dim, tmp_vector, eps);
        changed = ((changed) && (epsilon_indicator));
    }
    free(tmp_vector);
    return changed;
}

PyObject* array_to_PyList(Cluster *clusters, int k, int dim){
    Py_ssize_t i, j;
    PyObject *PyList = PyList_New(k);
    PyObject *item;
    for(i = 0; i < k; i++){
        item = PyList_New(dim);
        for(j = 0; j < dim; j++){
            PyList_SetItem(item, j, PyFloat_FromDouble(clusters[i].centroid[j]));
        }
        PyList_SetItem(PyList, i, item);
    }
    return PyList;
}


void kmeans(int max_iter, int n, int eps, Cluster* clusters, Point* points, int dim, int k) {
    int epsilon_check, iter, i, index;
    epsilon_check = 0;
    iter = 0;
    while ((iter < max_iter) && (1 - epsilon_check)) {
        for (i = 0; i < n; i++) {
            index = min_distance(&points[i], clusters, dim, k);
            add_point(&points[i], &clusters[index], dim);
        }
        epsilon_check = clusters_update(clusters, k, dim, eps);
        iter++;
    }
}


static PyObject* fit(PyObject* self, PyObject* args) {
    PyObject *final, *centroids_list, *points_list;
    Py_ssize_t k, n, dim, eps, max_iter;
    Point *points;
    Cluster *clusters;
    if (!PyArg_ParseTuple(args, "iiiiOO:fit", &k, &n, &dim, &eps, &max_iter, &centroids_list, &points_list)) {
        return NULL;
    }
    //list check
    points = allocate_mem((int)dim, (int)n);
    create_matrix(points_list, points, (int)dim, (int)n);
    createClusters((int)dim, centroids_list, (int)k);

    kmeans((int)max_iter, (int)n, eps, clusters, points, (int)dim, (int)k);
    final = array_to_PyList(clusters, (int)k, (int)dim);
    free_memory((int)k, (int)n, points, clusters);
    return final;
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
