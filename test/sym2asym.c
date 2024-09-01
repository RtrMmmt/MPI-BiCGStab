#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int row;
    int col;
    double value;
} MatrixEntry;

int compare(const void *a, const void *b) {
    MatrixEntry *entryA = (MatrixEntry *)a;
    MatrixEntry *entryB = (MatrixEntry *)b;

    if (entryA->row != entryB->row)
        return entryA->row - entryB->row;
    return entryA->col - entryB->col;
}

void add_upper_triangle_to_matrix(const char *input_file, const char *output_file) {
    FILE *input = fopen(input_file, "r");
    FILE *output = fopen(output_file, "w");

    if (!input || !output) {
        printf("Error opening file.\n");
        return;
    }

    char line[256];
    int nrows, ncols, nonzeros;
    MatrixEntry *entries = NULL;
    int entry_count = 0;
    int capacity = 0;

    // ヘッダと行列サイズの読み込み
    while (fgets(line, sizeof(line), input)) {
        if (line[0] == '%') {
            if (strstr(line, "symmetric")) {
                char *pos = strstr(line, "symmetric");
                strncpy(pos, "general  ", 9); // symmetric -> generalに変更
            }
            fprintf(output, "%s", line);
        } else {
            sscanf(line, "%d %d %d", &nrows, &ncols, &nonzeros);
            fprintf(output, "%d %d ", nrows, ncols);
            capacity = 2 * nonzeros; // 十分な容量を確保
            entries = (MatrixEntry *)malloc(capacity * sizeof(MatrixEntry));
            break;
        }
    }

    // エントリの読み込み
    while (fgets(line, sizeof(line), input)) {
        int row, col;
        double value;
        sscanf(line, "%d %d %lf", &row, &col, &value);

        if (entry_count == capacity) {
            capacity *= 2;
            entries = (MatrixEntry *)realloc(entries, capacity * sizeof(MatrixEntry));
        }

        entries[entry_count++] = (MatrixEntry){row, col, value};
        if (row != col) {
            entries[entry_count++] = (MatrixEntry){col, row, value};
        }
    }

    // エントリをソート
    qsort(entries, entry_count, sizeof(MatrixEntry), compare);

    fprintf(output, "%d\n", entry_count);
    for (int i = 0; i < entry_count; i++) {
        fprintf(output, "%d %d %lf\n", entries[i].row, entries[i].col, entries[i].value);
    }

    fclose(input);
    fclose(output);
    free(entries);
}

int main() {
    add_upper_triangle_to_matrix("s3dkq4m2.mtx", "output.mtx");
    return 0;
}
