
#include "stdafx.h"
#include <math.h>
#include <windows.h>

#include "neural_network.h"


__inline double f(double x)
{
	return tanh(x);
}

__inline double df(double x)
{
	return 1.0 -(pow(tanh(x), 2));
}

typedef struct _errors {
	double fd_error;
	double fd_moment_error;
	double quickprop_error;
} errors;


learning_selection * create_iris_selections(size_t count)
{
	learning_selection * selection;
	FILE *fp = fopen("iris.txt","r"); 
	
	if(fp) 
	{
		selection = (learning_selection *)calloc(count, sizeof(learning_selection));
		for(size_t i=0; i < count; i++)
		{
			selection[i].x = (double *)calloc(4, sizeof(double));
			selection[i].d = (double *)calloc(3, sizeof(double));
			selection[i].N = 4;
			selection[i].M = 3;
			fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf\n", &selection[i].x[0], &selection[i].x[1], &selection[i].x[2], &selection[i].x[3], 
				&selection[i].d[0], &selection[i].d[1], &selection[i].d[2]);
		}
		fclose(fp); 
	}
	return selection;
}


learning_selection * create_wine_selections(size_t count)
{
	learning_selection * selection;
	FILE *fp = fopen("wine.txt","r"); 

	if(fp) 
	{
		selection = (learning_selection *)calloc(count, sizeof(learning_selection));
		for(size_t i=0; i < count; i++)
		{
			selection[i].x = (double *)calloc(13, sizeof(double));
			selection[i].d = (double *)calloc(3, sizeof(double));
			selection[i].N = 13;
			selection[i].M = 3;
			fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &selection[i].x[0], &selection[i].x[1], &selection[i].x[2], &selection[i].x[3], 
				 &selection[i].x[4],  &selection[i].x[5],  &selection[i].x[6],  &selection[i].x[7],  &selection[i].x[8],  &selection[i].x[9],
				  &selection[i].x[10],  &selection[i].x[11],  &selection[i].x[12],
				&selection[i].d[0], &selection[i].d[1], &selection[i].d[2]);
		}
		fclose(fp); 
	}
	return selection;
}

void delete_selections(learning_selection * selections, size_t count)
{
	for(size_t i=0; i < count; i++)
	{
		free(selections[i].x);
		free(selections[i].d);
	}
	free(selections);
}

errors *errs;

void fd_error(size_t i, double error)
{
	errs[i].fd_error = error;
}

void fd_moment_error(size_t i, double error)
{
	errs[i].fd_moment_error = error;
}

void quickprop_error(size_t i, double error)
{
	errs[i].quickprop_error = error;
}

void print_errors(errors *errs, size_t count)
{
	for(size_t i = 0; i < count; i++)
	{
		printf("{\"it\": '%Iu', \"fd\": %f, \"fd_moment\": %f, \"quickprop\": %f},\n", i, errs[i].fd_error, errs[i].fd_moment_error, errs[i].quickprop_error);
	}
}



void iris_test()
{
	printf("[iris test]\n");
	neural_network *net = neural_network_new(4, 5, 3, f, df, sin, cos);
	init_random(net, -0.1, 0.1);
	learning_selection * selection = create_iris_selections(144);

	errs = (errors *)calloc(100, sizeof(errors));
	fd_teach(net, selection, 144, 0.01, 100, fd_error);
	quickprop_teach(net, selection, 144, 0.01, 100, quickprop_error);
	fd_moment_teach(net, selection, 144, 0.01, 0.6, 100, fd_moment_error);

	print_errors(errs, 100);

	double x1[] = {5.1, 3.5, 1.4, 0.3};
	double x2[] = {6.4, 2.9, 4.3, 1.3};
	double x3[] = {6.1, 3.0, 4.9, 1.8};

	double x4[] = {5.7, 3.8, 1.7, 0.3};
	double x5[] = {6.6, 3.0, 4.4, 1.4};
	double x6[] = {6.4, 2.8, 5.6, 2.1};

	printf("\n");
	set_x(net, x1, 4);
	calc_y(net);
	print_y(net);
	printf("\n");

	set_x(net, x2, 4);
	calc_y(net);
	print_y(net);
	printf("\n");

	set_x(net, x3, 4);
	calc_y(net);
	print_y(net);
	printf("\n");

	set_x(net, x4, 4);
	calc_y(net);
	print_y(net);
	printf("\n");

	set_x(net, x5, 4);
	calc_y(net);
	print_y(net);
	printf("\n");

	set_x(net, x6, 4);
	calc_y(net);
	print_y(net);
	printf("\n");

	delete_selections(selection, 144);
	neural_network_delete(net);
}



void wine_test()
{
	printf("[wine test]\n");
	neural_network *net = neural_network_new(13, 13, 3, f, df, sin, cos);
	init_random(net, -0.1, 0.1);
	learning_selection * selection = create_wine_selections(90);

	errs = (errors *)calloc(100, sizeof(errors));
	fd_teach(net, selection, 90, 0.02, 100, fd_error);
	quickprop_teach(net, selection, 90, 0.02, 100, quickprop_error);
	fd_moment_teach(net, selection, 90, 0.02, 0.8, 100, fd_moment_error);
	
	print_errors(errs, 100);

	double x1[] = {0.8, 0.152, 0.725, 0.221052632, 0.456521739, 0.756666667, 0.678, 0.34, 0.4925, 0.479166667, 0.525, 0.616666667, 0.835948645};
	double x2[] = {0.3425, 0.034, 0.46, 0.452631579, 0.086956522, 0.37, 0.4, 0.27, 0.26, 0.306666667, 0.56, 0.826666667, 0.165477889};
	double x3[] = {0.4425, 0.278, 0.64, 0.447368421, 0.173913043, 0.13, 0.102, 0.48, 0.16, 0.741666583, 0.285, 0.21, 0.136947218};

	printf("\n");
	set_x(net, x1, 13);
	calc_y(net);
	print_y(net);
	printf("\n");
	set_x(net, x2, 13);
	calc_y(net);
	print_y(net);
	printf("\n");
	set_x(net, x3, 13);
	calc_y(net);
	print_y(net);
	printf("\n");

	delete_selections(selection, 90);
	neural_network_delete(net);
}

double measure_time(void (*func)(), size_t count)
{
	LARGE_INTEGER timerFrequency, timerStart, timerStop;
	double t = 0.0;

	QueryPerformanceFrequency(&timerFrequency);
	QueryPerformanceCounter(&timerStart);
	for(size_t i = 0; i< count; i++)
	{
		func();
	}
	QueryPerformanceCounter(&timerStop);
	t = (double)( timerStop.QuadPart -	timerStart.QuadPart ) / timerFrequency.QuadPart;
	return t;
}


int _tmain(int argc, _TCHAR* argv[])
{
	double t1,t2;
	t1 = measure_time(iris_test, 1);
	t2 = measure_time(wine_test, 1);

	printf("time: %f s\n", t1);
	printf("time: %f s\n", t2);

	system("pause");
	return 0;
}