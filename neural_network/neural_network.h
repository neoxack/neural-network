#pragma once

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

typedef struct _neural_network {
	double *x;
	double *v;
	double *y;

	double **w1;
	double **w2;

	double **w1_random;
	double **w2_random;

	double **w1_old;
	double **w2_old;

	size_t N;
	size_t K;
	size_t M;

	double (*f1)(double);
	double (*df1)(double);

	double (*f2)(double);
	double (*df2)(double);

	double *sums1;
	double *sums2;

	double *df_u1;
	double *df_u2;
	double *error;
	double **dE_dw1;
	double **dE_dw2;

	//quickprop
	double **dE_dw1_old;
	double **dE_dw2_old;

} neural_network;

typedef struct _learning_selection {
	double *x;
	size_t N;
	double *d;
	size_t M;
} learning_selection;

neural_network * neural_network_new(size_t N, size_t K, size_t M, double (*f1)(double), double (*df1)(double), double (*f2)(double), double (*df2)(double))
{
	neural_network *net = (neural_network *)calloc(1, sizeof(neural_network));
	srand((unsigned int)time(0));

	net->N = N;
	net->K = K;
	net->M = M;

	net->f1 = f1;
	net->df1 = df1;

	net->f2 = f2;
	net->df2 = df2;

	net->x = (double *)calloc(N+1, sizeof(double));
	net->v = (double *)calloc(K+1, sizeof(double));
	net->y = (double *)calloc(M+1, sizeof(double));
	net->error = (double *)calloc(M+1, sizeof(double));

	net->sums1 = (double *)calloc(K+1, sizeof(double));
	net->sums2 = (double *)calloc(M+1, sizeof(double));

	net->df_u1 = (double *)calloc(K+1, sizeof(double));
	net->df_u2 = (double *)calloc(M+1, sizeof(double));

	net->x[0] = 1.0;
	net->v[0] = 1.0;

	net->w1 = (double **)malloc((K+1) * sizeof(double *));
	net->w1_random = (double **)malloc((K+1) * sizeof(double *));
	net->w1_old = (double **)malloc((K+1) * sizeof(double *));
	net->dE_dw1 = (double **)malloc((K+1) * sizeof(double *));
	net->dE_dw1_old = (double **)malloc((K+1) * sizeof(double *));
	for (size_t i = 0; i <= K; i++) 
	{
		net->w1[i] = (double *)malloc((N+1) * sizeof(double));
		net->w1_random[i] = (double *)calloc((N+1) , sizeof(double));
		net->w1_old[i] = (double *)calloc((N+1) , sizeof(double));
		net->dE_dw1[i] = (double *)calloc((N+1) , sizeof(double));
		net->dE_dw1_old[i] = (double *)calloc((N+1), sizeof(double));
	}

	net->w2 = (double **)malloc((M+1) * sizeof(double *));
	net->w2_random = (double **)malloc((M+1) * sizeof(double *));
	net->w2_old = (double **)malloc((M+1) * sizeof(double *));
	net->dE_dw2 = (double **)malloc((M+1) * sizeof(double *));
	net->dE_dw2_old = (double **)malloc((M+1) * sizeof(double *));
	for (size_t i = 0; i <= M; i++)
	{
		net->w2[i] = (double *)malloc((K+1) * sizeof(double));
		net->w2_random[i] = (double *)malloc((K+1) * sizeof(double));
		net->w2_old[i] = (double *)calloc((K+1), sizeof(double));
		net->dE_dw2[i] = (double *)calloc((K+1) , sizeof(double));
		net->dE_dw2_old[i] = (double *)calloc((K+1), sizeof(double));
	}

	return net;
}

void neural_network_delete(neural_network * net)
{
	free(net->x);
	free(net->v);
	free(net->y);
	free(net->error);

	free(net->sums1);
	free(net->sums2);

	free(net->df_u1);
	free(net->df_u2);

	for (size_t i = 0; i <= net->K; i++) 
	{
		free(net->w1[i]);
		free(net->w1_random[i]);
		free(net->w1_old[i]);
		free(net->dE_dw1[i]);
		free(net->dE_dw1_old[i]);
	}

	for (size_t i = 0; i <= net->M; i++)
	{
		free(net->w2[i]);
		free(net->w2_random[i]);
		free(net->w2_old[i]);
		free(net->dE_dw2[i]);
		free(net->dE_dw2_old[i]);
	}

	free(net->w1);
	free(net->w1_random);
	free(net->w1_old);
	free(net->dE_dw1);
	free(net->dE_dw1_old);

	free(net->w2);
	free(net->w2_random);
	free(net->w2_old);
	free(net->dE_dw2);
	free(net->dE_dw2_old);

	free(net);
}

static double fRand(double min, double max)
{
	return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
}

void init_random(neural_network * net, double a, double b)
{
	for (size_t i = 1; i <= net->K; i++)
		for (size_t j = 0; j <= net->N; j++)
			net->w1_random[i][j] = fRand(a, b);

	for (size_t i = 1; i <= net->M; i++)
		for (size_t j = 0; j <= net->K; j++)
			net->w2_random[i][j] = fRand(a, b);
}

void set_random_weights(neural_network * net)
{
	for (size_t i = 1; i <= net->K; i++)
		for (size_t j = 0; j <= net->N; j++)
			net->w1[i][j] = net->w1_random[i][j];
			
	for (size_t i = 1; i <= net->M; i++)
		for (size_t j = 0; j <= net->K; j++)
			net->w2[i][j] = net->w2_random[i][j];
}

void set_x(neural_network * net, double *x, size_t count)
{
	if(count == net->N)
	{
		net->x[0] = 1.0;
		memcpy(&net->x[1], x, count * sizeof(double));
	}
}

__inline static void calc_v(neural_network * net)
{
	double sum;
	net->v[0] = 1.0;
	for(size_t i = 1; i <= net->K; i++)
	{
		sum = 0.0;
		for(size_t j = 0; j <= net->N; j++)
		{
			sum += (net->w1[i][j] * net->x[j]);
		}
		net->sums1[i] = sum;
		net->v[i] = net->f1(sum);
	}
}

void calc_y(neural_network * net)
{
	calc_v(net);
	double sum;
	for(size_t s = 1; s <= net->M; s++)
	{
		sum = 0.0;
		for(size_t i = 0; i <= net->K; i++)
		{
			sum += (net->w2[s][i] * net->v[i]);
		}
		net->sums2[s] = sum;
		net->y[s] = net->f2(sum);
	}
}

void print_y(neural_network * net)
{
	for(size_t s = 1; s <= net->M; s++)
	{
		printf("%f\n", net->y[s]);
	}
}

static int round(double number)
{
	return (number >= 0) ? 1 : -1;
}

void print_round_y(neural_network * net)
{
	for(size_t s = 1; s <= net->M; s++)
	{
		printf("%d\n", round(net->y[s]));
	}
}

__inline static double back_propagation_step(neural_network * net, learning_selection *selection)
{
	double *d = selection->d - 1;
	double E = 0.0;
	set_x(net, selection->x, selection->N);
	calc_y(net);
	
	//calc df/du2 and errors
	for(size_t s = 1; s <= net->M; s++)
	{
		net->df_u2[s] = net->df2(net->sums2[s]);
		net->error[s] = net->y[s] - d[s];
		E+=pow(net->error[s], 2);
	}
	//calc df/du1

	for(size_t i = 1; i <= net->K; i++)
	{
		net->df_u1[i] = net->df1(net->sums1[i]);
	}

	//calc dE/dw2
	for(size_t s = 1; s <= net->M; s++)
	{
		for(size_t i = 0; i <= net->K; i++)
		{
			net->dE_dw2[s][i] = net->error[s] * net->df_u2[s] * net->v[i];
		}
	}
	//calc dE/dw1
	double sum;
	for(size_t i = 1; i <= net->K; i++)
	{
		sum = 0.0;
		for(size_t s = 1; s <= net->M; s++)
		{
			sum += (net->error[s] * net->df_u2[s] * net->w2[s][i]);
		}

		for(size_t j = 0; j <= net->N; j++)
		{	
			net->dE_dw1[i][j] = sum * net->df_u1[i] * net->x[j];
		}
	}
	return E;
}

__inline static void change_weights(neural_network * net, double η) 
{
	for (size_t i = 1; i <= net->K; i++)
		for (size_t j = 0; j <= net->N; j++)
			net->w1[i][j] -= (η * net->dE_dw1[i][j]);

	for (size_t i = 1; i <= net->M; i++)
		for (size_t j = 0; j <= net->K; j++)
			net->w2[i][j] -= (η * net->dE_dw2[i][j]);
}

__inline static void change_weights_with_moments(neural_network * net, double η, double α) 
{
	double w_new;

	for (size_t i = 1; i <= net->K; i++)
		for (size_t j = 0; j <= net->N; j++)
		{
			w_new = net->w1[i][j] - (η * net->dE_dw1[i][j] + α*(net->w1[i][j] - net->w1_old[i][j]));
			net->w1_old[i][j] = net->w1[i][j];
			net->w1[i][j] = w_new;
		}

	for (size_t i = 1; i <= net->M; i++)
		for (size_t j = 0; j <= net->K; j++)
		{
			w_new = net->w2[i][j] - (η * net->dE_dw2[i][j] + α*(net->w2[i][j] - net->w2_old[i][j]));
			net->w2_old[i][j] = net->w2[i][j];
			net->w2[i][j] = w_new;
		}
}


//fd simple
static void fd_teach(neural_network * net, learning_selection *selections, size_t selection_count, double η, double retries_count, void(*error_callback)(size_t it, double e) )
{
	double eta = η;
	double E;
	set_random_weights(net);

	for(size_t r = 0; r < retries_count; r++)
	{	
		E = 0.0;
		for(size_t i = 0; i < selection_count; i++)
		{
			E+=back_propagation_step(net, &selections[i]);
			change_weights(net,  eta);
		}
		eta *= 0.98;
		error_callback(r, E/2);
	}
}

static void fd_teach(neural_network * net, learning_selection *selections, size_t selection_count, double η, double retries_count)
{
	double eta = η;
	double E;
	set_random_weights(net);

	for(size_t r = 0; r < retries_count; r++)
	{	
		E = 0.0;
		for(size_t i = 0; i < selection_count; i++)
		{
			E+=back_propagation_step(net, &selections[i]);
			change_weights(net,  eta);
		}
		eta *= 0.98;
	}
}

//fd with moment
static void fd_moment_teach(neural_network * net, learning_selection *selections, size_t selection_count, double η, double α, double retries_count, void(*error_callback)(size_t it, double e))
{
	double eta = η;
	double E;
	set_random_weights(net);

	for(size_t r = 0; r < retries_count; r++)
	{	
		E = 0.0;
		for(size_t i = 0; i < selection_count; i++)
		{
			E+=back_propagation_step(net, &selections[i]);
			change_weights_with_moments(net, eta, α);
		}
		error_callback(r, E/2);
	}
}

static void fd_moment_teach(neural_network * net, learning_selection *selections, size_t selection_count, double η, double α, double retries_count)
{
	double eta = η;
	double E;
	set_random_weights(net);

	for(size_t r = 0; r < retries_count; r++)
	{	
		E = 0.0;
		for(size_t i = 0; i < selection_count; i++)
		{
			E+=back_propagation_step(net, &selections[i]);
			change_weights_with_moments(net, eta, α);
		}
	}
}

double alpha_max = 1.75;

void quickprop_change_weights(neural_network * net, double η)
{
	double delta_w;
	double epsilon = η;
	double slope;
	double prev_step;
	double shrink_factor = (double) (alpha_max / (1.0 + alpha_max));
	double prev_slope;

	for (size_t i = 1; i <= net->K; i++)
		for (size_t j = 0; j <= net->N; j++)
		{
			prev_step = net->w1[i][j] - net->w1_old[i][j];
			slope = -net->dE_dw1[i][j];
			prev_slope = net->dE_dw1_old[i][j];
			delta_w = 0.0;

			if(prev_step > 0.001)
			{
				if(slope > (shrink_factor * prev_slope))
					delta_w += alpha_max * prev_step;	
				else
					delta_w+= prev_step * slope / (prev_slope - slope);
			}
			else if(prev_step < -0.001)
			{
				if(slope < (shrink_factor * prev_slope))
					delta_w += alpha_max * prev_step;	
				else
					delta_w += prev_step * slope / (prev_slope - slope);	
			}
			else 
				delta_w += epsilon * slope; 

			net->w1_old[i][j] = net->w1[i][j];
			net->w1[i][j] +=delta_w;
			net->dE_dw1_old[i][j] = slope;
		}

	for (size_t i = 1; i <= net->M; i++)
		for (size_t j = 0; j <= net->K; j++)
		{
			prev_step = net->w2[i][j] - net->w2_old[i][j];
			slope = -net->dE_dw2[i][j];
			prev_slope = net->dE_dw2_old[i][j];
			delta_w = 0.0;

			if(prev_step > 0.001)
			{
				if(slope > (shrink_factor * prev_slope))
					delta_w +=alpha_max * prev_step;	
				else
					delta_w+= prev_step * slope / (prev_slope - slope);	
			}
			else if(prev_step < -0.001)
			{
				if(slope < (shrink_factor * prev_slope))
					delta_w += alpha_max * prev_step;	
				else
					delta_w += prev_step * slope / (prev_slope - slope);	
			}
			else 
				delta_w += epsilon * slope; 

			net->w2_old[i][j] = net->w2[i][j];
			net->w2[i][j] +=delta_w;
			net->dE_dw2_old[i][j] = slope;
		}

}



//quickprop
void quickprop_teach(neural_network * net, learning_selection *selections, size_t selection_count, double η, double retries_count, void(*error_callback)(size_t it, double e))
{
	double E;
	double eta = η;
	set_random_weights(net);
	for(size_t r = 0; r < retries_count; r++)
	{	
		E = 0.0;
		for(size_t i = 0; i < selection_count; i++)
		{
			E+=back_propagation_step(net, &selections[i]);
			quickprop_change_weights(net, eta);		
		}

		error_callback(r, E/2);	
	}
}

void quickprop_teach(neural_network * net, learning_selection *selections, size_t selection_count, double η, double retries_count)
{
	double E;
	double eta = η;
	set_random_weights(net);
	for(size_t r = 0; r < retries_count; r++)
	{	
		E = 0.0;
		for(size_t i = 0; i < selection_count; i++)
		{
			E+=back_propagation_step(net, &selections[i]);
			quickprop_change_weights(net, eta);		
		}
	}
}