/* kmeans_1d_openacc_fp16.c
   K-means 1D (C99 + OpenACC), implementação "naive" para GPU com FP16.
   - Usa o tipo de dado __half para precisão de 16 bits.
   - Lê X (N linhas, 1 coluna) e C_init (K linhas, 1 coluna) de CSVs sem cabeçalho.
   - Itera assignment + update até max_iter ou variação relativa do SSE < eps.
   - Salva (opcional) assign (N linhas) e centróides finais (K linhas).

   Compilar com NVIDIA HPC SDK:
   nvc -acc -O2 -lm -gpu=cc89 kmeans_1d_openacc_fp16.c -o kmeans_1d_openacc_fp16
   Uso:      ./kmeans_1d_openacc_fp16 dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

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

static _Float16 *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    _Float16 *A = (_Float16*)malloc((size_t)R * sizeof(_Float16));
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
        // Lê como float e converte para _Float16
        A[r] = (_Float16)strtof(tok, NULL);
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

static void write_centroids_csv(const char *path, const _Float16 *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) {
        // Converte _Float16 para float para imprimir
        fprintf(f, "%.6f\n", (float)C[c]);
    }
    fclose(f);
}

/* ---------- k-means 1D ---------- */
static _Float16 assignment_step_1d(const _Float16 *X, const _Float16 *C, int *assign, int N, int K){
    _Float16 sse = (_Float16)0.0f;
    #pragma acc kernels loop reduction(+:sse) present(X, C, assign)
    for(int i=0;i<N;i++){
        int best = -1;
        _Float16 bestd = (_Float16)FLT_MAX;
        for(int c=0;c<K;c++){
            _Float16 diff = X[i] - C[c];
            _Float16 d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

static void update_step_1d(const _Float16 *X, _Float16 *C, const int *assign, int N, int K, float *sum, int *cnt){
    #pragma acc kernels loop present(sum, cnt)
    for(int c=0; c<K; c++) {
        sum[c] = 0.0f;
        cnt[c] = 0;
    }

    // Loop 1: Calcula a contagem de pontos por cluster (atomic em int)
    #pragma acc kernels loop present(assign, cnt)
    for(int i=0;i<N;i++){
        int a = assign[i];
        #pragma acc atomic update
        cnt[a] += 1;
    }

    // Loop 2: Calcula a soma dos valores dos pontos por cluster (atomic em float)
    #pragma acc kernels loop present(X, assign, sum)
    for(int i=0;i<N;i++){
        int a = assign[i];
        #pragma acc atomic update
        sum[a] += (float)X[i];
    }

    #pragma acc kernels loop present(X, C, sum, cnt)
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = (_Float16)(sum[c] / (float)cnt[c]);
        else           C[c] = X[0];
    }
}

static void kmeans_1d(const _Float16 *X, _Float16 *C, int *assign,
                      int N, int K, int max_iter, _Float16 eps,
                      int *iters_out, _Float16 *sse_out)
{
    _Float16 prev_sse = (_Float16)FLT_MAX;
    _Float16 sse = (_Float16)0.0f;
    int it;

    float *sum = (float*)calloc((size_t)K, sizeof(float));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    #pragma acc data copyin(X[0:N]) copy(C[0:K]) create(assign[0:N], sum[0:K], cnt[0:K])
    {
        for(it=0; it<max_iter; it++){
            sse = assignment_step_1d(X, C, assign, N, K);
            
            float rel_f = fabsf((float)sse - (float)prev_sse);
            float prev_sse_f = (float)prev_sse;
            rel_f /= (prev_sse_f > 0.0f ? prev_sse_f : 1.0f);

            if(rel_f < (float)eps){ it++; break; }
            
            update_step_1d(X, C, assign, N, K, sum, cnt);
            prev_sse = sse;
        }
    }

    free(sum); free(cnt);
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    _Float16 eps   = (argc>4)? (_Float16)strtof(argv[4], NULL) : (_Float16)1e-4f;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    if(max_iter <= 0 || (float)eps <= 0.0f){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    _Float16 *X = read_csv_1col(pathX, &N);
    _Float16 *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    clock_t t0 = clock();
    int iters = 0; _Float16 sse = (_Float16)0.0f;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D (OpenACC FP16)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, (float)eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, (float)sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}
