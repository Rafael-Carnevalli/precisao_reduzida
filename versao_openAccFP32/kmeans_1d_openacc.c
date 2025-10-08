/* kmeans_1d_openacc.c
   K-means 1D (C99 + OpenACC), implementação "naive" para GPU com FP32.
   - Lê X (N linhas, 1 coluna) e C_init (K linhas, 1 coluna) de CSVs sem cabeçalho.
   - Itera assignment + update até max_iter ou variação relativa do SSE < eps.
   - Salva (opcional) assign (N linhas) e centróides finais (K linhas).

   Compilar com NVIDIA HPC SDK: nvc -acc -O2 -lm kmeans_1d_openacc.c -o kmeans_1d_openacc
   Uso:      ./kmeans_1d_openacc dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h> // Para FLT_MAX

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static float *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    float *A = (float*)malloc((size_t)R * sizeof(float));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = strtof(tok, NULL);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const float *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- k-means 1D ---------- */
/* assignment: para cada X[i], encontra c com menor (X[i]-C[c])^2 */
static float assignment_step_1d(const float *X, const float *C, int *assign, int N, int K){
    float sse = 0.0f;
    #pragma acc kernels loop reduction(+:sse) present(X, C, assign)
    for(int i=0;i<N;i++){
        int best = -1;
        float bestd = FLT_MAX;
        for(int c=0;c<K;c++){
            float diff = X[i] - C[c];
            float d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

/* update: média dos pontos de cada cluster (1D) */
static void update_step_1d(const float *X, float *C, const int *assign, int N, int K, float *sum, int *cnt){
    // Zera os acumuladores na GPU
    #pragma acc kernels loop present(sum, cnt)
    for(int c=0; c<K; c++) {
        sum[c] = 0.0f;
        cnt[c] = 0;
    }

    // Calcula soma e contagem de cada cluster de forma atômica
    #pragma acc kernels loop present(X, assign, sum, cnt)
    for(int i=0;i<N;i++){
        int a = assign[i];
        #pragma acc atomic update
        cnt[a] += 1;
        #pragma acc atomic update
        sum[a] += X[i];
    }

    // Atualiza os centróides
    #pragma acc kernels loop present(X, C, sum, cnt)
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (float)cnt[c];
        else           C[c] = X[0];
    }
}

static void kmeans_1d(const float *X, float *C, int *assign,
                      int N, int K, int max_iter, float eps,
                      int *iters_out, float *sse_out)
{
    float prev_sse = FLT_MAX;
    float sse = 0.0f;
    int it;

    // Aloca memória para os arrays auxiliares do passo de update
    float *sum = (float*)calloc((size_t)K, sizeof(float));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    // Cria a região de dados na GPU: X é copiado para a GPU, C é copiado nos dois sentidos,
    // e os outros arrays são alocados na GPU.
    #pragma acc data copyin(X[0:N]) copy(C[0:K]) create(assign[0:N], sum[0:K], cnt[0:K])
    {
        for(it=0; it<max_iter; it++){
            sse = assignment_step_1d(X, C, assign, N, K);
            
            float rel = fabsf(sse - prev_sse) / (prev_sse > 0.0f ? prev_sse : 1.0f);
            if(rel < eps){ it++; break; }
            
            update_step_1d(X, C, assign, N, K, sum, cnt);
            prev_sse = sse;
        }
    } // Fim da região de dados, C é copiado de volta para a CPU

    free(sum); free(cnt);
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
// Ponto de entrada do programa
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    float eps   = (argc>4)? strtof(argv[4], NULL) : 1e-4f;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    if(max_iter <= 0 || eps <= 0.0f){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    float *X = read_csv_1col(pathX, &N);
    float *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    clock_t t0 = clock();
    int iters = 0; float sse = 0.0f;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D (OpenACC)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}
