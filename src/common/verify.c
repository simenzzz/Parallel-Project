#include <stdio.h>
#include <stdlib.h>

static long file_size(FILE *fp)
{
    long size;

    if (fseek(fp, 0L, SEEK_END) != 0) {
        return -1L;
    }
    size = ftell(fp);
    if (size < 0) {
        return -1L;
    }
    if (fseek(fp, 0L, SEEK_SET) != 0) {
        return -1L;
    }
    return size;
}

int main(int argc, char **argv)
{
    FILE *fa;
    FILE *fb;
    long size_a;
    long size_b;
    int mismatches = 0;
    const int max_print = 16;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <grid_a> <grid_b>\n", argv[0]);
        return 1;
    }

    fa = fopen(argv[1], "rb");
    if (fa == NULL) {
        perror(argv[1]);
        return 1;
    }
    fb = fopen(argv[2], "rb");
    if (fb == NULL) {
        perror(argv[2]);
        fclose(fa);
        return 1;
    }

    size_a = file_size(fa);
    size_b = file_size(fb);
    if (size_a < 0 || size_b < 0) {
        fprintf(stderr, "Failed to stat input files\n");
        fclose(fa);
        fclose(fb);
        return 1;
    }
    if (size_a != size_b) {
        fprintf(stderr, "Size mismatch: %s=%ld bytes, %s=%ld bytes\n",
                argv[1], size_a, argv[2], size_b);
        fclose(fa);
        fclose(fb);
        return 1;
    }

    for (long i = 0; i < size_a; ++i) {
        int a = fgetc(fa);
        int b = fgetc(fb);
        if (a == EOF || b == EOF) {
            fprintf(stderr, "Unexpected EOF while comparing\n");
            fclose(fa);
            fclose(fb);
            return 1;
        }
        if (a != b) {
            if (mismatches < max_print) {
                fprintf(stderr, "Mismatch at byte %ld: %d != %d\n", i, a, b);
            }
            ++mismatches;
        }
    }

    fclose(fa);
    fclose(fb);

    if (mismatches != 0) {
        fprintf(stderr, "Found %d mismatches\n", mismatches);
        return 1;
    }

    puts("Voxel grids are identical");
    return 0;
}
