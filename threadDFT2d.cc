// Threaded two-dimensional Discrete FFT Transform
// Michael Martin
// ECE8893 Project 2

#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>

#include "Complex.h"
#include "InputImage.h"

#define NUM_THREADS 16
#define TWO_PI (M_PI * 2)

using namespace std;

// You will likely need global variables indicating how
// many threads there are, and a Complex* that points to the
// 2d image being Transformed.
InputImage *image;
Complex *data;
unsigned N;
string file;
void (*Transform)(Complex *h);

// barrier variables
pthread_mutex_t barrierMutex;
int barrierThreads, barrierCount;
bool *threadFlag, barrierFlag;

// pthread condition
pthread_cond_t exitCond;
pthread_mutex_t exitMutex;
pthread_mutex_t activeMutex;
int activeThreads;

//==================================================================
// Reordering Functions
//==================================================================

// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D Transform.
unsigned ReverseBits(unsigned v)
{ //  Provided to students
    unsigned n = N; // Size of array (which is even 2 power k value)
    unsigned r = 0; // Return value

    for (--n; n > 0; n >>= 1)
    {
        r <<= 1;        // Shift return value
        r |= (v & 0x1); // Merge in next bit
        v >>= 1;        // Shift reversal value
    }
    return r;
}

// Function that reorders a row for the FFT algorithm
void ReorderRow(Complex *H)
{
    bool swapped[N];
    for (unsigned i = 0; i < N; i++) {
        swapped[i] = false;
    }
    for (unsigned i = 0; i < N; i++) {
        if (!swapped[i]) {
            unsigned j = ReverseBits(i);
            swap(H[i], H[j]);
            swapped[i] = true;
            swapped[j] = true;
        }
    }
}

//==================================================================
// Barrier Functions
//==================================================================

// GRAD Students implement the following 2 functions.
// Undergrads can use the built-in barriers in pthreads.

// Call MyBarrier_Init once in main
void MyBarrier_Init(int numThreads)
{
    barrierThreads = numThreads;
    barrierCount = numThreads;
    pthread_mutex_init(&barrierMutex, 0);
    threadFlag = new bool[numThreads];
    for (int i = 0; i < numThreads; ++i) {
        threadFlag[i] = true;
    }
    barrierFlag = true;
}

// Called by MyBarrier
int FetchAndDecrementCount()
{
    pthread_mutex_lock(&barrierMutex);
    int count = barrierCount--;
    pthread_mutex_unlock(&barrierMutex);
    return count;
}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier(int threadId)
{
    threadFlag[threadId] = !threadFlag[threadId];
    if (FetchAndDecrementCount() == 1) {
        barrierCount = barrierThreads;
        barrierFlag = threadFlag[threadId];
    } else {
        while (barrierFlag != threadFlag[threadId]);
    }
}

//==================================================================
// Transform Functions
//==================================================================

void Transpose(Complex *data)
{
    for (unsigned i = 0; i < N; i++ ) {
        for (unsigned j = i + 1; j < N; j++ ) {
            swap(data[i * N + j], data[j * N + i]);
        }
    }
}

void ComputeWeights(Complex *W)
{
    for (unsigned n = 0; n < (N >> 1); n++) {
        double exponent = TWO_PI * n / N;
        W[n] = Complex(cos(exponent), -sin(exponent));
    }
}

void FFT(Complex *H)
{
    // Reorder row
    ReorderRow(H);

    // Precompute weights
    Complex W[N >> 1];
    ComputeWeights(W);

    for (unsigned x = 2; x <= N; x <<= 1) {
        unsigned xdiv2 = (x >> 1);
        // i is the offset (0, x, 2x, ..., N - x)
        for (unsigned i = 0; i < N; i += x) {
            // k is the index (0, 1, 2, ..., x/2 - 1)
            for (unsigned k = 0; k < xdiv2; k++) {
                unsigned j = k + i;
                Complex even = H[j];
                Complex odd = W[k * N / x] * H[j + xdiv2];
                H[j] = even + odd;
                H[j + xdiv2] = even - odd;
            }
        }
    }
}

void IFFT(Complex* H)
{
    // Reorder row
    ReorderRow(H);

    // Precompute weights
    Complex W[N >> 1];
    ComputeWeights(W);

    // imag * (-1)
    for (unsigned i = 0; i < N; i++) {
        H[i].imag = -H[i].imag;
    }

    for (unsigned x = 2; x <= N; x <<= 1) {
        unsigned xdiv2 = (x >> 1);
        // i is the offset (0, x, 2x, ..., N - x)
        for (unsigned i = 0; i < N; i += x) {
            // k is the index (0, 1, 2, ..., x/2 - 1)
            for (unsigned k = 0; k < xdiv2; k++) {
                unsigned j = k + i;
                Complex even = H[j];
                Complex odd = W[k * N / x] * H[j + xdiv2];
                H[j] = even + odd;
                H[j + xdiv2] = even - odd;
            }
        }
    }

    // imag * (-1) and divide by N
    for (unsigned i = 0; i < N; i++) {
        H[i].imag = -H[i].imag;
        H[i] = H[i] * (1.0 / N);
    }
}

//==================================================================
// Main Functions
//==================================================================

void *Transform2DThread(void *v)
{   // This is the thread starting point.  "v" is the thread number

    unsigned long threadId = (unsigned long)v;
    // cout << "Thread " << threadId << " starting first transform.\n";

    // Calculate 1d DFT for assigned rows
    for (unsigned i = threadId; i < N; i += NUM_THREADS) {
        Transform(data + (i * N));
    }

    // wait for all to complete rows
    // cout << "Thread " << threadId << " entering barrier.\n";
    MyBarrier(threadId);

    // Save data and transpose
    if (threadId == 0) {
        if (file != "") {
            image->SaveImageData(file.c_str(), data, N, N);
        }
        Transpose(data);
    }

    MyBarrier(threadId);
    // cout << "Thread " << threadId << " starting second transform.\n";

    // Calculate 1d DFT for assigned columns
    for (unsigned i = threadId; i < N; i += NUM_THREADS) {
        Transform(data + (i * N));
    }

    // Decrement active count and signal main if all complete
    // cout << "Thread " << threadId << " exiting.\n";
    pthread_mutex_lock(&activeMutex);
    activeThreads--;
    if (activeThreads == 0) { // Last to exit, notify main
        // cout << "Thread " << threadId << " signaling exit condition.\n";
        pthread_mutex_unlock(&activeMutex);
        pthread_mutex_lock(&exitMutex);
        pthread_cond_signal(&exitCond);
        pthread_mutex_unlock(&exitMutex);
        // cout << "Thread " << threadId << " exit condition signaled.\n";
    } else {
        pthread_mutex_unlock(&activeMutex);
    }

    return 0;
}

void Transform2D(const char *inputFN) 
{   // Do the 2D Transform here.

    // Create the helper object for reading the image
    image = new InputImage(inputFN);

    // Create the global pointer to the image array data
    data = image->GetImageData();
    N = image->GetWidth();
    file = "MyAfter1D.txt";
    Transform = &FFT;

    // Create 16 threads
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], 0, &Transform2DThread,  (void *)i);
    }

    // Wait for all threads to finish
    pthread_cond_wait(&exitCond, &exitMutex);
    Transpose(data);
    image->SaveImageData("MyAfter2D.txt", data, N, N);

    // Reinitialize
    file = "";
    Transform = &IFFT;
    activeThreads = NUM_THREADS;
    pthread_mutex_init(&activeMutex, 0);
    pthread_mutex_init(&exitMutex, 0);
    pthread_cond_init(&exitCond, 0);
    pthread_mutex_lock(&exitMutex);

    // Start 16 threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], 0, &Transform2DThread,  (void *)i);
    }

    // Wait for all threads to finish
    pthread_cond_wait(&exitCond, &exitMutex);
    Transpose(data);
    image->SaveImageData("MyAfterInverse.txt", data, N, N);
}

int main(int argc, char **argv)
{
    string fn("Tower.txt"); // default file name
    if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line

    // Initialize barrier and condition
    MyBarrier_Init(NUM_THREADS);
    activeThreads = NUM_THREADS;
    pthread_mutex_init(&activeMutex, 0);
    pthread_mutex_init(&exitMutex, 0);
    pthread_cond_init(&exitCond, 0);
    pthread_mutex_lock(&exitMutex);

    // Perform the Transform.
    Transform2D(fn.c_str());

    // Cleanup
    delete image;
    delete[] threadFlag;
}
