
 //Write a C program for implementing process creationand termination.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    printf("Sajan Bista\n");
   // Declare a variable to hold the process ID
   pid_t pid;

   // Call fork() to create a new process
   pid = fork();

   if (pid < 0) {
       // Error occurred
       perror("ERROR: Failed to create child process");
       return 1;
   } else if (pid == 0) {
       // This block executes in the child process
       printf("This is the child process. PID: %d, Parent PID: %d\n", getpid(), getppid());
   } else {
       // This block executes in the parent process
       printf("This is the parent process. PID: %d\n", getpid());
   }

   // Both parent and child will execute this part
   printf("Exiting the program. PID: %d\n", getpid());
   return 0;
}


/*
//Write a C program for implementing thread creation and
//termination.
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// Thread function
void* threadFunction(void* arg) {
    int thread_id = *(int*)arg;
    printf("Thread %d: Hello, I am a new thread!\n", thread_id);
    return NULL; // Terminates the thread
}

int main(void) {
    printf("Sajan Bista");
    int n; // Number of threads
    printf("Enter the number of threads to create: ");
    scanf("%d", &n);

    pthread_t threads[n]; // Array to hold thread IDs
    int thread_args[n];   // Array to hold thread arguments

    // Create threads
    for (int i = 0; i < n; i++) {
        thread_args[i] = i + 1; // Thread argument (thread ID)
        if (pthread_create(&threads[i], NULL, threadFunction, &thread_args[i]) != 0) {
            perror("ERROR: Thread creation failed");
            exit(EXIT_FAILURE);
        }
        printf("Main: Created thread %d\n", i + 1);
    }

    // Wait for threads to finish
    for (int i = 0; i < n; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("ERROR: Thread join failed");
            exit(EXIT_FAILURE);
        }
        printf("Main: Thread %d has finished execution.\n", i + 1);
    }

    printf("Main: All threads have completed. Exiting program.\n");
    return 0;
}




#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h> // Include for wait()

#define SHM_SIZE 1024 // Size of shared memory

int main(void) {
    printf("Sajan Bista\n");
    int shmid;
    char *shared_memory;
    pid_t pid;

    // Create a shared memory segment
    shmid = shmget(IPC_PRIVATE, SHM_SIZE, IPC_CREAT | 0666);
    if (shmid == -1) {
        perror("shmget");
        exit(EXIT_FAILURE);
    }

    // Attach the shared memory segment
    shared_memory = shmat(shmid, NULL, 0);
    if (shared_memory == (char *)-1) {
        perror("shmat");
        exit(EXIT_FAILURE);
    }

    // Create a child process
    pid = fork();
    if (pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) { // Child process
        // Read message from shared memory
        printf("Child process read: %s\n", shared_memory);
        shmdt(shared_memory); // Detach shared memory
    } else { // Parent process
        const char *message = "Hello from parent in shared memory!";
        strcpy(shared_memory, message); // Write message to shared memory
        wait(NULL); // Wait for child process to finish
        shmdt(shared_memory); // Detach shared memory
        shmctl(shmid, IPC_RMID, NULL); // Remove shared memory
    }

    return 0;
}
//Simulate IPC techniques using C program.


#include <stdio.h>

struct Process
{
    int id, arrival, burst, completion, waiting, turnaround;
};

void sortByArrival(struct Process proc[], int n)
{
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (proc[j].arrival > proc[j + 1].arrival)
            {
                struct Process temp = proc[j];
                proc[j] = proc[j + 1];
                proc[j + 1] = temp;
            }
}

void calculateTimes(struct Process proc[], int n)
{
    proc[0].completion = proc[0].arrival + proc[0].burst;
    for (int i = 1; i < n; i++)
    {
        proc[i].completion = proc[i].arrival > proc[i - 1].completion ? proc[i].arrival + proc[i].burst : proc[i - 1].completion + proc[i].burst;
    }

    for (int i = 0; i < n; i++)
    {
        proc[i].turnaround = proc[i].completion - proc[i].arrival;
        proc[i].waiting = proc[i].turnaround - proc[i].burst;
    }
}

void printResults(struct Process proc[], int n)
{
    int totalWaiting = 0, totalTurnaround = 0;
    printf("\nProcess\t  Arrival Time\t  Burst Time\tCompletion Time\t  Waiting Time\tTurnaround Time\n");
    for (int i = 0; i < n; i++)
    {
        totalWaiting += proc[i].waiting;
        totalTurnaround += proc[i].turnaround;
        printf("%d\t\t%d\t\t%d\t\t%d\t\t%d\t\t%d\n", proc[i].id, proc[i].arrival, proc[i].burst, proc[i].completion, proc[i].waiting, proc[i].turnaround);
    }
    printf("\nAverage waiting time = %.2f", (float)totalWaiting / n);
    printf("\nAverage turnaround time = %.2f\n", (float)totalTurnaround / n);
}

int main(void)
{
    printf("Sajan Bista\n");
    int n;
    printf("Enter the number of processes: ");
    scanf("%d", &n);

    struct Process proc[n];
    for (int i = 0; i < n; i++)
    {
        proc[i].id = i + 1;
        printf("Enter arrival time and burst time for process %d: ", proc[i].id);
        scanf("%d %d", &proc[i].arrival, &proc[i].burst);
    }

    sortByArrival(proc, n);
    calculateTimes(proc, n);
    printResults(proc, n);

    return 0;
}


//SJF (Shortest Job First)
#include <stdio.h>
#include <stdbool.h>

#define MAX_PROCESSES 100

// Structure to represent a process
typedef struct {
    int id;                // Process ID
    int burstTime;        // Burst Time
    int arrivalTime;      // Arrival Time
    int completionTime;   // Completion Time
    int turnaroundTime;   // Turnaround Time
    int waitingTime;      // Waiting Time
} Process;

// Function to calculate TAT and WT
void calculateTimes(Process processes[], int n) {
    for (int i = 0; i < n; i++) {
        processes[i].turnaroundTime = processes[i].completionTime - processes[i].arrivalTime;
        processes[i].waitingTime = processes[i].turnaroundTime - processes[i].burstTime;
    }
}

// Function to print process details
void printProcessDetails(Process processes[], int n) {
    printf("\nProcess ID\tBurst Time\tArrival Time\tCompletion Time\tTAT\tWT\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t\t%d\t\t%d\t\t%d\t\t%d\t%d\n",
               processes[i].id,
               processes[i].burstTime,
               processes[i].arrivalTime,
               processes[i].completionTime,
               processes[i].turnaroundTime,
               processes[i].waitingTime);
    }
}

// SJF Scheduling Algorithm
void sjf(Process processes[], int n) {
    bool completed[MAX_PROCESSES] = {false}; // Track completed processes
    int completedCount = 0, currentTime = 0;

    while (completedCount < n) {
        int shortestIndex = -1;
        int shortestTime = 9999999;

        // Find the process with the shortest burst time
        for (int i = 0; i < n; i++) {
            if (!completed[i] && processes[i].arrivalTime <= currentTime) {
                if (processes[i].burstTime < shortestTime) {
                    shortestTime = processes[i].burstTime;
                    shortestIndex = i;
                }
            }
        }

        if (shortestIndex != -1) {
            // Process the shortest job
            currentTime += processes[shortestIndex].burstTime;
            processes[shortestIndex].completionTime = currentTime;
            completed[shortestIndex] = true; // Mark as completed
            completedCount++;
        } else {
            // If no process is ready to execute, move time forward
            currentTime++;
        }
    }

    calculateTimes(processes, n);
}

// Main function
int main(void) {
    printf("Sajan Bista\n");
    Process processes[MAX_PROCESSES];
    int n;

    printf("Enter the number of processes: ");
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        processes[i].id = i + 1; // Assign Process IDs starting from 1
        printf("Enter burst time and arrival time for Process %d: ", i + 1);
        scanf("%d %d", &processes[i].burstTime, &processes[i].arrivalTime);
    }

    // SJF
    printf("\nShortest Job First Scheduling:\n");
    sjf(processes, n);
    printProcessDetails(processes, n);

    return 0;
}


 //SRTF (Shortest Remaining Time First)

 #include <stdio.h>
 #include <limits.h>

 // Structure to represent a process
 struct Process {
     int id, arrival, burst, remaining, completion, waiting, turnaround;
 };

 // Function to calculate times using SRTF
 void calculateTimes(struct Process proc[], int n) {
     int currentTime = 0, completed = 0, shortest = -1, minRemainingTime;
     int isProcessFound = 0;

     // Initialize remaining times for all processes
     for (int i = 0; i < n; i++) {
         proc[i].remaining = proc[i].burst;
     }

     while (completed < n) {
         // Find the process with the shortest remaining time
         minRemainingTime = INT_MAX;
         isProcessFound = 0;

         for (int i = 0; i < n; i++) {
             if (proc[i].arrival <= currentTime && proc[i].remaining > 0 && proc[i].remaining < minRemainingTime) {
                 minRemainingTime = proc[i].remaining;
                 shortest = i;
                 isProcessFound = 1;
             }
         }

         if (!isProcessFound) {
             // No process is ready, increment the time
             currentTime++;
             continue;
         }

         // Process the selected process for 1 unit of time
         proc[shortest].remaining--;
         currentTime++;

         // If the process is completed
         if (proc[shortest].remaining == 0) {
             proc[shortest].completion = currentTime;
             proc[shortest].turnaround = proc[shortest].completion - proc[shortest].arrival;
             proc[shortest].waiting = proc[shortest].turnaround - proc[shortest].burst;
             completed++;
         }
     }
 }

 // Function to print process details
 void printResults(struct Process proc[], int n) {
     int totalWaiting = 0, totalTurnaround = 0;

     printf("\nProcess\t  Arrival Time\t  Burst Time\tCompletion Time\t  Waiting Time\tTurnaround Time\n");
     for (int i = 0; i < n; i++) {
         totalWaiting += proc[i].waiting;
         totalTurnaround += proc[i].turnaround;
         printf("%d\t\t%d\t\t%d\t\t%d\t\t%d\t\t%d\n", proc[i].id, proc[i].arrival, proc[i].burst, proc[i].completion, proc[i].waiting, proc[i].turnaround);
     }

     printf("\nAverage waiting time = %.2f", (float)totalWaiting / n);
     printf("\nAverage turnaround time = %.2f\n", (float)totalTurnaround / n);
 }

 int main(void) {
     printf("Sajan Bista\n");
     int n;

     printf("Enter the number of processes: ");
     scanf("%d", &n);

     struct Process proc[n];
     for (int i = 0; i < n; i++) {
         proc[i].id = i + 1;
         printf("Enter arrival time and burst time for process %d: ", proc[i].id);
         scanf("%d %d", &proc[i].arrival, &proc[i].burst);
     }

     calculateTimes(proc, n);
     printResults(proc, n);

     return 0;
 }

// RR (Round-Robbin CPU Scheduling Algorithm)

#include <stdio.h>

// Structure to represent a process
struct Process {
    int id, arrival, burst, remaining, completion, waiting, turnaround;
};

// Function to calculate Round-Robin scheduling
void calculateTimes(struct Process proc[], int n, int quantum) {
    int currentTime = 0, completed = 0;
    int queue[n * 2], front = 0, rear = 0; // Double size queue for safety
    int visited[n]; // Track if a process is added to the queue

    for (int i = 0; i < n; i++) {
        visited[i] = 0;
        proc[i].remaining = proc[i].burst;
    }

    // Add the first process to the queue
    queue[rear++] = 0;
    visited[0] = 1;

    while (completed < n) {
        if (front == rear) { // If the queue is empty, advance time to the next arrival
            for (int i = 0; i < n; i++) {
                if (!visited[i] && proc[i].arrival > currentTime) {
                    currentTime = proc[i].arrival;
                    queue[rear++] = i;
                    visited[i] = 1;
                    break;
                }
            }
        }

        int i = queue[front++ % (n * 2)]; // Get the next process from the queue

        if (proc[i].remaining > quantum) {
            currentTime += quantum;
            proc[i].remaining -= quantum;
        } else {
            currentTime += proc[i].remaining;
            proc[i].remaining = 0;
            proc[i].completion = currentTime;
            proc[i].turnaround = proc[i].completion - proc[i].arrival;
            proc[i].waiting = proc[i].turnaround - proc[i].burst;
            completed++;
        }

        // Add newly arrived processes to the queue
        for (int j = 0; j < n; j++) {
            if (proc[j].arrival <= currentTime && proc[j].remaining > 0 && !visited[j]) {
                queue[rear++ % (n * 2)] = j;
                visited[j] = 1;
            }
        }

        // Re-add the current process to the queue if it's not yet completed
        if (proc[i].remaining > 0) {
            queue[rear++ % (n * 2)] = i;
        }
    }
}

// Function to print process details
void printResults(struct Process proc[], int n) {
    int totalWaiting = 0, totalTurnaround = 0;

    printf("\nProcess\tArrival Time\tBurst Time\tCompletion Time\tWaiting Time\tTurnaround Time\n");
    for (int i = 0; i < n; i++) {
        totalWaiting += proc[i].waiting;
        totalTurnaround += proc[i].turnaround;
        printf("%d\t\t%d\t\t%d\t\t%d\t\t%d\t\t%d\n", proc[i].id, proc[i].arrival, proc[i].burst, proc[i].completion, proc[i].waiting, proc[i].turnaround);
    }

    printf("\nAverage waiting time = %.2f", (float)totalWaiting / n);
    printf("\nAverage turnaround time = %.2f\n", (float)totalTurnaround / n);
}

int main(void) {
    int n, quantum;

    printf("Enter the number of processes: ");
    scanf("%d", &n);

    struct Process proc[n];
    for (int i = 0; i < n; i++) {
        proc[i].id = i + 1;
        printf("Enter arrival time and burst time for process %d: ", proc[i].id);
        scanf("%d %d", &proc[i].arrival, &proc[i].burst);
    }

    printf("Enter the time quantum: ");
    scanf("%d", &quantum);

    calculateTimes(proc, n, quantum);
    printResults(proc, n);

    return 0;
}



//Priority Scheduling. Also compute TAT and WT time
//of each algorithm.

#include <stdio.h>
#include <stdbool.h>

struct Process {
  int id, arrival, burst, priority, completion, waiting, turnaround;
};

// Function to sort processes based on arrival time and priority
void sortByPriority(struct Process proc[], int n) {
  for (int i = 0; i < n - 1; i++) {
      for (int j = 0; j < n - i - 1; j++) {
          if ((proc[j].arrival > proc[j + 1].arrival) ||
              (proc[j].arrival == proc[j + 1].arrival && proc[j].priority > proc[j + 1].priority)) {
              struct Process temp = proc[j];
              proc[j] = proc[j + 1];
              proc[j + 1] = temp;
          }
      }
  }
}

// Function to calculate times
void calculateTimes(struct Process proc[], int n) {
  int completed = 0;
  int currentTime = 0;
  bool isCompleted[n];

  // Initialize completed array
  for (int i = 0; i < n; i++) {
      isCompleted[i] = false;
  }

  while (completed < n) {
      int idx = -1;
      // Find the process with the highest priority among the processes that have arrived
      for (int i = 0; i < n; i++) {
          if (proc[i].arrival <= currentTime && !isCompleted[i]) {
              if (idx == -1 || proc[i].priority < proc[idx].priority) {
                  idx = i; // Update index for the highest priority process
              }
          }
      }

      // If no process is found, increment the time
      if (idx == -1) {
          currentTime++;
          continue;
      }

      // Process the selected process
      currentTime += proc[idx].burst;
      proc[idx].completion = currentTime;
      proc[idx].turnaround = proc[idx].completion - proc[idx].arrival;
      proc[idx].waiting = proc[idx].turnaround - proc[idx].burst;
      isCompleted[idx] = true; // Mark the process as completed
      completed++; // Increment completed process count
  }
}

// Function to print results
void printResults(struct Process proc[], int n) {
  int totalWaiting = 0, totalTurnaround = 0;

  printf("\nProcess\tArrival Time\tBurst Time\tPriority\tCompletion Time\tWaiting Time\tTurnaround Time\n");
  for (int i = 0; i < n; i++) {
      totalWaiting += proc[i].waiting;
      totalTurnaround += proc[i].turnaround;
      printf("%d\t%d\t\t%d\t\t%d\t\t%d\t\t%d\t\t%d\n",
              proc[i].id, proc[i].arrival, proc[i].burst, proc[i].priority,
              proc[i].completion, proc[i].waiting, proc[i].turnaround);
  }

  printf("\nAverage waiting time = %.2f", (float)totalWaiting / n);
  printf("\nAverage turnaround time = %.2f\n", (float)totalTurnaround / n);
}

int main(void) {
    printf("Sajan Bista \n");
  int n;

  printf("Enter the number of processes: ");
  scanf("%d", &n);

  struct Process proc[n];
  for (int i = 0; i < n; i++) {
      proc[i].id = i + 1;
      printf("Enter arrival time, burst time and priority for process %d: ", proc[i].id);
      scanf("%d %d %d", &proc[i].arrival, &proc[i].burst, &proc[i].priority);
  }

  // Sort processes based on arrival time and priority
  sortByPriority(proc, n);
  calculateTimes(proc, n);
  printResults(proc, n);

  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define BUFFER_SIZE 5
#define NUM_ITEMS 5

int buffer[BUFFER_SIZE];
int in = 0;
int out = 0;

sem_t empty;       // Semaphore to track empty slots
sem_t full;        // Semaphore to track full slots
pthread_mutex_t mutex; // Mutex for critical section

// Producer function
void* producer(void* arg) {
    for (int i = 0; i < NUM_ITEMS; i++) {
        sem_wait(&empty);               // Wait for an empty slot
        pthread_mutex_lock(&mutex);     // Enter critical section

        buffer[in] = i;                 // Add item to buffer
        printf("Producer produced: %d\n", i);
        in = (in + 1) % BUFFER_SIZE;    // Increment and wrap-around index

        pthread_mutex_unlock(&mutex);  // Exit critical section
        sem_post(&full);               // Signal a full slot

        sleep(1); // Simulate time taken to produce an item
    }
    return NULL;
}

// Consumer function
void* consumer(void* arg) {
    for (int i = 0; i < NUM_ITEMS; i++) {
        sem_wait(&full);                // Wait for a full slot
        pthread_mutex_lock(&mutex);     // Enter critical section

        int item = buffer[out];         // Remove item from buffer
        printf("Consumer consumed: %d\n", item);
        out = (out + 1) % BUFFER_SIZE;  // Increment and wrap-around index

        pthread_mutex_unlock(&mutex);  // Exit critical section
        sem_post(&empty);              // Signal an empty slot

        sleep(1); // Simulate time taken to consume an item
    }
    return NULL;
}

int main(void) {
    printf("Sajan Bista\n");
    pthread_t prod_thread, cons_thread;

    // Initialize semaphores
    sem_init(&empty, 0, BUFFER_SIZE); // Initially, all slots are empty
    sem_init(&full, 0, 0);           // Initially, no slots are full
    pthread_mutex_init(&mutex, NULL); // Initialize mutex

    // Create producer and consumer threads
    pthread_create(&prod_thread, NULL, producer, NULL);
    pthread_create(&cons_thread, NULL, consumer, NULL);

    // Wait for threads to finish
    pthread_join(prod_thread, NULL);
    pthread_join(cons_thread, NULL);

    // Clean up resources
    sem_destroy(&empty);
    sem_destroy(&full);
    pthread_mutex_destroy(&mutex);

    return 0;
}



//To implement deadlock detection Algorithms. (Bankers Algorithms)

#include <stdio.h>
#include <stdbool.h>

#define MAX_PROCESSES 10
#define MAX_RESOURCES 10

int n, m; // n = number of processes, m = number of resource types
int available[MAX_RESOURCES], max[MAX_PROCESSES][MAX_RESOURCES], allocation[MAX_PROCESSES][MAX_RESOURCES], need[MAX_PROCESSES][MAX_RESOURCES];

void calculateNeed() {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            need[i][j] = max[i][j] - allocation[i][j];
        }
    }
}

bool isSafe() {
    int work[MAX_RESOURCES], finish[MAX_PROCESSES] = {0};
    int safeSequence[MAX_PROCESSES], index = 0;
    int i, j, k;

    // Initialize work with available resources
    for (i = 0; i < m; i++) {
        work[i] = available[i];
    }

    // Try to find a safe sequence
    for (k = 0; k < n; k++) {
        bool found = false;
        for (i = 0; i < n; i++) {
            if (finish[i] == 0) { // If process i is not finished
                for (j = 0; j < m; j++) {
                    if (need[i][j] > work[j]) {
                        break; // Process i cannot proceed
                    }
                }
                
                // If all resources for process i can be allocated
                if (j == m) {
                    // Allocate resources to process i
                    for (int y = 0; y < m; y++) {
                        work[y] += allocation[i][y]; // Add allocated resources to work
                    }
                    safeSequence[index++] = i; // Record the process in safe sequence
                    finish[i] = 1; // Mark process as finished
                    found = true; // Indicate that we found a process
                }
            }
        }

        if (!found) { // No process could proceed
            break; // Exit the loop if no process can proceed
        }
    }

    // Check if all processes finished
    for (i = 0; i < n; i++) {
        if (finish[i] == 0) {
            printf("The system is in an unsafe state!\n");
            return false;
        }
    }

    // If the system is safe
    printf("The system is in a safe state.\nSafe sequence is: ");
    for (i = 0; i < index; i++) { // Print only the finished processes
        printf("P%d ", safeSequence[i]);
    }
    printf("\n");
    return true;
}

int main() {
    printf("Sajan Bista\n");
    printf("Enter the number of processes (max %d): ", MAX_PROCESSES);
    scanf("%d", &n);
    printf("Enter the number of resource types (max %d): ", MAX_RESOURCES);
    scanf("%d", &m);

    // Input for available resources
    printf("Enter the available resources: \n");
    for (int i = 0; i < m; i++) {
        scanf("%d", &available[i]);
    }

    // Input for maximum resource matrix
    printf("Enter the maximum resource matrix (Max): \n");
    for (int i = 0; i < n; i++) {
        printf("For process P%d: ", i);
        for (int j = 0; j < m; j++) {
            scanf("%d", &max[i][j]);
        }
    }

    // Input for allocation resource matrix
    printf("Enter the allocation resource matrix (Allocation): \n");
    for (int i = 0; i < n; i++) {
        printf("For process P%d: ", i);
        for (int j = 0; j < m; j++) {
            scanf("%d", &allocation[i][j]);
        }
    }

    // Calculate need and check if the system is in a safe state
    calculateNeed();
    isSafe();

    return 0;
}


//7 a, first fit


#include <stdio.h>

void firstFit(int blockSize[], int m, int processSize[], int n) {
    int allocation[n];
    int i, j; // Declare variables outside the for loop for older C standards

    // Initialize all allocations to -1 (indicating no allocation yet)
    for (i = 0; i < n; i++) {
        allocation[i] = -1;
    }

    // Loop through each process and find the first block that fits
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (blockSize[j] >= processSize[i]) {
                allocation[i] = j; // Allocate block j to process i
                blockSize[j] -= processSize[i]; // Reduce available memory in this block
                break;
            }
        }
    }

    // Print the results
    printf("\nProcess No.\tProcess Size\tBlock No.\n");
    for (i = 0; i < n; i++) {
        printf("%d\t\t  %d\t\t", i + 1, processSize[i]);
        if (allocation[i] != -1)
            printf("  %d\n", allocation[i] + 1);
        else
            printf("Need to wait\n");
    }
}

int main() {
    printf("Sajan Bista\n");
    int m, n, i;

    printf("Enter the number of memory blocks: ");
    scanf("%d", &m);
    int blockSize[m];
    printf("Enter the sizes of the memory blocks in order: ");
    for (i = 0; i < m; i++) {
        scanf("%d", &blockSize[i]);
    }

    printf("Enter the number of processes: ");
    scanf("%d", &n);
    int processSize[n];
    printf("Enter the sizes of the processes in order: ");
    for (i = 0; i < n; i++) {
        scanf("%d", &processSize[i]);
    }

    firstFit(blockSize, m, processSize, n);

    return 0;
}

 
// linked list

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Define a structure for a memory block
typedef struct Block {
    int id;                  // Block ID
    bool is_free;           // Free status (true = free, false = allocated)
    struct Block* next;     // Pointer to the next block
} Block;

// Head of the linked list
Block* head = NULL;

// Function to initialize the linked list with memory blocks
void initialize_memory(int num_blocks) {
    for (int i = 0; i < num_blocks; i++) {
        Block* new_block = (Block*)malloc(sizeof(Block));
        new_block->id = i;
        new_block->is_free = true;  // Mark as free
        new_block->next = head;      // Link new block to the list
        head = new_block;            // Update head to new block
    }
}

// Function to allocate a block
int allocate_block() {
    Block* temp = head;
    while (temp != NULL) {
        if (temp->is_free) {
            temp->is_free = false;  // Mark as allocated
            printf("Block %d allocated.\n", temp->id);
            return temp->id;        // Return block ID
        }
        temp = temp->next;          // Move to the next block
    }
    printf("No free blocks available.\n");
    return -1; // No free block found
}

// Function to free a block
void free_block(int block_id) {
    Block* temp = head;
    while (temp != NULL) {
        if (temp->id == block_id) {
            temp->is_free = true;   // Mark as free
            printf("Block %d freed.\n", temp->id);
            return;
        }
        temp = temp->next;          // Move to the next block
    }
    printf("Invalid block ID: %d\n", block_id);
}

// Function to display the memory state
void display_memory() {
    Block* temp = head;
    printf("Memory State:\n");
    while (temp != NULL) {
        printf("Block %d: Status = %s\n", temp->id, temp->is_free ? "Free" : "Allocated");
        temp = temp->next;          // Move to the next block
    }
}

int main() {
    printf("Sajan Bista\n");
    int num_blocks = 5;

    // Initialize memory blocks
    initialize_memory(num_blocks);
    display_memory();

    // Allocate some blocks
    allocate_block();
    allocate_block();
    display_memory();

    // Free a block
    free_block(0);  // Free the first block
    display_memory();

    return 0;
}


//worstfit

#include <stdio.h>

void worstFit(int blockSize[], int m, int processSize[], int n) {
    int allocation[n];

    // Initialize all allocations to -1 (indicating no allocation yet)
    for (int i = 0; i < n; i++) {
        allocation[i] = -1;
    }

    // Loop through each process to find the worst block that fits
    for (int i = 0; i < n; i++) {
        int worstIdx = -1;
        for (int j = 0; j < m; j++) {
            if (blockSize[j] >= processSize[i]) {
                if (worstIdx == -1 || blockSize[worstIdx] < blockSize[j]) {
                    worstIdx = j;
                }
            }
        }

        // If we found a block for the process
        if (worstIdx != -1) {
            allocation[i] = worstIdx; // Allocate block to process
            blockSize[worstIdx] -= processSize[i]; // Reduce available memory in this block
        }
    }

    // Print the results
    printf("\nProcess No.\tProcess Size\tBlock No.\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t\t%d\t\t", i + 1, processSize[i]);
        if (allocation[i] != -1)
            printf("%d\n", allocation[i] + 1);
        else
            printf("Need to wait\n");
    }
}

int main() {
    printf("Sajan Bista");
    int m, n;

    printf("Enter the number of memory blocks: ");
    scanf("%d", &m);
    int blockSize[m];
    printf("Enter the sizes of the memory blocks in order: ");
    for (int i = 0; i < m; i++) {
        scanf("%d", &blockSize[i]);
    }

    printf("Enter the number of processes: ");
    scanf("%d", &n);
    int processSize[n];
    printf("Enter the sizes of the processes in order: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &processSize[i]);
    }

    worstFit(blockSize, m, processSize, n);

    return 0;
}

//a. Bit Vector


#include <stdio.h>  // For printf
#include <stdbool.h> // For bool, true, false

#define BLOCKS 16  // Number of blocks

// Initialize bitmap with all blocks free (0)
unsigned char bitmap[BLOCKS / 8] = {0};

// Function to set the bit at index in the bitmap
void set_bit(int index) {
    bitmap[index / 8] |= (1 << (index % 8));
}

// Function to clear the bit at index in the bitmap
void clear_bit(int index) {
    bitmap[index / 8] &= ~(1 << (index % 8));
}

// Function to check if the bit at index is set
bool is_bit_set(int index) {
    return (bitmap[index / 8] & (1 << (index % 8))) != 0;
}

// Function to allocate a block
int allocate_block() {
    int i; // Declare variable outside the loop
    for (i = 0; i < BLOCKS; i++) {
        if (!is_bit_set(i)) {
            set_bit(i);
            return i;  // Return the index of the allocated block
        }
    }
    return -1;  // No free blocks available
}

// Function to free a block
void free_block(int index) {
    if (index >= 0 && index < BLOCKS) {
        clear_bit(index);
    }
}

int main() {
    printf("Sajan Bista");
    // Allocate some blocks
    int block1 = allocate_block();
    int block2 = allocate_block();

    if (block1 != -1) {
        printf("Allocated block %d\n", block1);
    } else {
        printf("Failed to allocate block 1\n");
    }

    if (block2 != -1) {
        printf("Allocated block %d\n", block2);
    } else {
        printf("Failed to allocate block 2\n");
    }

    // Free a block
    if (block1 != -1) {
        free_block(block1);
        printf("Freed block %d\n", block1);
    }

    

    return 0;
}

// linked list

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Define a structure for a memory block
typedef struct Block {
    int id;                  // Block ID
    bool is_free;           // Free status (true = free, false = allocated)
    struct Block* next;     // Pointer to the next block
} Block;

// Head of the linked list
Block* head = NULL;

// Function to initialize the linked list with memory blocks
void initialize_memory(int num_blocks) {
    for (int i = 0; i < num_blocks; i++) {
        Block* new_block = (Block*)malloc(sizeof(Block));
        new_block->id = i;
        new_block->is_free = true;  // Mark as free
        new_block->next = head;      // Link new block to the list
        head = new_block;            // Update head to new block
    }
}

// Function to allocate a block
int allocate_block() {
    Block* temp = head;
    while (temp != NULL) {
        if (temp->is_free) {
            temp->is_free = false;  // Mark as allocated
            printf("Block %d allocated.\n", temp->id);
            return temp->id;        // Return block ID
        }
        temp = temp->next;          // Move to the next block
    }
    printf("No free blocks available.\n");
    return -1; // No free block found
}

// Function to free a block
void free_block(int block_id) {
    Block* temp = head;
    while (temp != NULL) {
        if (temp->id == block_id) {
            temp->is_free = true;   // Mark as free
            printf("Block %d freed.\n", temp->id);
            return;
        }
        temp = temp->next;          // Move to the next block
    }
    printf("Invalid block ID: %d\n", block_id);
}

// Function to display the memory state
void display_memory() {
    Block* temp = head;
    printf("Memory State:\n");
    while (temp != NULL) {
        printf("Block %d: Status = %s\n", temp->id, temp->is_free ? "Free" : "Allocated");
        temp = temp->next;          // Move to the next block
    }
}

int main() {
    printf("Sajan Bista\n");
    int num_blocks = 5;

    // Initialize memory blocks
    initialize_memory(num_blocks);
    display_memory();

    // Allocate some blocks
    allocate_block();
    allocate_block();
    display_memory();

    // Free a block
    free_block(0);  // Free the first block
    display_memory();

    return 0;
}


//9a fifo

#include <stdio.h>

int main() {
    printf("sajan Bista\n");
    int capacity, n, page_faults = 0, page_hits = 0, front = 0;

    printf("Enter the number of frames: ");
    scanf("%d", &capacity);  // Number of frames available (capacity of memory)

    printf("Enter the number of page requests: ");
    scanf("%d", &n);  // Number of pages (sequence length)

    // Declare frames and pages arrays
    int frames[100];   // Frames to hold the pages (Assuming a max capacity of 100)
    int pages[100];    // Array for page requests (Assuming a max number of page requests of 100)
    int filled = 0;    // Number of pages currently filled in frames
    int i, j;          // Loop variables

    // Initialize all frames to -1 (indicating they are empty)
    for (i = 0; i < capacity; i++) {
        frames[i] = -1;
    }

    // Input the page sequence
    printf("Enter the page reference string: ");
    for (i = 0; i < n; i++) {
        scanf("%d", &pages[i]);
    }

    // Iterate over each page in the reference string
    for (i = 0; i < n; i++) {
        int page = pages[i];
        int found = 0;  // Flag to check if the page is already in frame

        // Check if the page is already in one of the frames (page hit)
        for (j = 0; j < filled; j++) {
            if (frames[j] == page) {
                found = 1;
                break;
            }
        }

        // Display the current page request
        printf("Page %2d: ", page);

        // If the page is found, it's a hit
        if (found) {
            page_hits++;
        } else {
            // If the page is not found, it's a fault
            if (filled < capacity) {
                // If there is still space in the frames
                frames[filled] = page;
                filled++;
            } else {
                // Replace the oldest page (FIFO)
                frames[front] = page;
                front = (front + 1) % capacity;  // Move to next position in FIFO
            }
            page_faults++;
        }

        // Display the current state of the frames with uniform spacing
        for (j = 0; j < capacity; j++) {
            if (frames[j] == -1)
                printf("  -  ");
            else
                printf("%3d  ", frames[j]);
        }

        // Indicate hit or fault
        if (found) {
            printf("[ H ]\n");
        } else {
            printf("[ F ]\n");
        }
    }

    // Output the total number of page faults and hits
    printf("\nTotal Page Faults: %d\n", page_faults);
    printf("Total Page Hits: %d\n", page_hits);

    return 0;
}




//9b lru

#include <stdio.h>

struct Page {
    int value;     // Page value
    int frequency; // Frequency of usage
    int last_used; // Last used time for LFU tie-breaking
};

int main() {
    int capacity, n, page_faults = 0, page_hits = 0, time = 0;
    
    // Get the number of frames
    printf("Enter the number of frames: ");
    scanf("%d", &capacity);  // Number of frames available (capacity of memory)

    // Get the number of page requests
    printf("Enter the number of page requests: ");
    scanf("%d", &n);  // Number of pages (sequence length)

    // Declare arrays with fixed sizes
    struct Page frames[100];   // Frames to hold the pages and their frequency (assuming a max of 100 frames)
    int pages[100];            // Array for page requests (assuming a max of 100 page requests)

    // Initialize frames (set page value to -1 and frequency to 0)
    int i;  // Loop variable
    for (i = 0; i < capacity; i++) {
        frames[i].value = -1;
        frames[i].frequency = 0;
        frames[i].last_used = 0;
    }

    // Input the page sequence
    printf("Enter the page reference string: ");
    for (i = 0; i < n; i++) {
        scanf("%d", &pages[i]);
    }

    // Iterate over each page in the reference string
    for (i = 0; i < n; i++) {
        int page = pages[i];
        int found = 0;  // Flag to check if the page is already in frame
        time++;         // Update the time for LFU tie-breaking

        // Check if the page is already in one of the frames (page hit)
        int j;  // Loop variable for frame checking
        for (j = 0; j < capacity; j++) {
            if (frames[j].value == page) {
                found = 1;
                frames[j].frequency++;   // Increment frequency on hit
                frames[j].last_used = time;  // Update the last used time
                break;
            }
        }

        // Display the current page request
        printf("Page %2d: ", page);

        // If page is found, it's a hit
        if (found) {
            page_hits++;
        } else {
            // If page is not found, it's a fault
            page_faults++;

            // Find the frame to replace using LFU logic
            int min_freq = frames[0].frequency, replace_idx = 0;
            for (j = 1; j < capacity; j++) {
                // Replace the least frequently used page or in case of tie, the least recently used page
                if (frames[j].frequency < min_freq ||
                   (frames[j].frequency == min_freq && frames[j].last_used < frames[replace_idx].last_used)) {
                    min_freq = frames[j].frequency;
                    replace_idx = j;
                }
            }

            // Replace the page in the chosen frame
            frames[replace_idx].value = page;
            frames[replace_idx].frequency = 1;  // Initialize frequency to 1
            frames[replace_idx].last_used = time;  // Update the last used time
        }

        // Display the current state of the frames with uniform spacing
        for (j = 0; j < capacity; j++) {
            if (frames[j].value == -1)
                printf("  -  ");
            else
                printf("%3d  ", frames[j].value);
        }

        // Indicate hit or fault
        if (found) {
            printf("[ H ]\n");
        } else {
            printf("[ F ]\n");
        }
    }

    // Output the total number of page faults and hits
    printf("\nTotal Page Faults: %d\n", page_faults);
    printf("Total Page Hits: %d\n", page_hits);

    return 0;
}




//9c op

#include <stdio.h>

int find_farthest(int pages[], int frames[], int n, int current_index, int capacity) {
    int farthest_index = -1;
    int farthest_distance = -1;
    int i, j; // Declare loop variables here

    for (i = 0; i < capacity; i++) {
        // Check how far this page is used in the future
        for (j = current_index + 1; j < n; j++) {
            if (frames[i] == pages[j]) {
                if (j > farthest_distance) {
                    farthest_distance = j;
                    farthest_index = i;
                }
                break; // Break if the page is found
            }
        }

        // If the page is not used in the future at all
        if (j == n) {
            return i;  // Replace this page since it's never used again
        }
    }

    return (farthest_index == -1) ? 0 : farthest_index;
}

int main() {
    int capacity, n, page_faults = 0, page_hits = 0;
    int filled = 0; // Number of pages in frames currently filled

    // Get the number of frames
    printf("Enter the number of frames: ");
    scanf("%d", &capacity);  // Number of frames available (capacity of memory)

    // Get the number of page requests
    printf("Enter the number of page requests: ");
    scanf("%d", &n);  // Number of pages (sequence length)

    // Declare arrays with fixed sizes
    int frames[100];   // Frames to hold the pages (assuming a max of 100 frames)
    int pages[100];    // Array for page requests (assuming a max of 100 page requests)

    // Initialize all frames to -1 (indicating they are empty)
    int i; // Declare loop variable
    for (i = 0; i < capacity; i++) {
        frames[i] = -1;
    }

    // Input the page sequence
    printf("Enter the page reference string: ");
    for (i = 0; i < n; i++) {
        scanf("%d", &pages[i]);
    }

    // Iterate over each page in the reference string
    for (i = 0; i < n; i++) {
        int page = pages[i];
        int found = 0;  // Flag to check if page is already in frame
        int j; // Declare j here for the loop

        // Check if the page is already in one of the frames (page hit)
        for (j = 0; j < filled; j++) {
            if (frames[j] == page) {
                found = 1;
                break;
            }
        }

        // Display the current page request
        printf("Page %2d: ", page);

        // If page is found, it's a hit
        if (found) {
            page_hits++;
        } else {
            // If page is not found, it's a fault
            page_faults++;
            if (filled < capacity) {
                // If there is still space in the frames
                frames[filled] = page;
                filled++;
            } else {
                // Find the page that is used farthest in the future and replace it
                int farthest_index = find_farthest(pages, frames, n, i, capacity);
                frames[farthest_index] = page;
            }
        }

        // Display the current state of the frames with uniform spacing
        for (j = 0; j < capacity; j++) {
            if (frames[j] == -1)
                printf("  -  ");
            else
                printf("%3d  ", frames[j]);
        }

        // Indicate hit or fault
        if (found) {
            printf("[ H ]\n");
        } else {
            printf("[ F ]\n");
        }
    }

    // Output the total number of page faults and hits
    printf("\nTotal Page Faults: %d\n", page_faults);
    printf("Total Page Hits: %d\n", page_hits);

    return 0;
}

//9d

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

// Function to find an element in the queue
// Returns the index if found, -1 otherwise
int findPage(int *q, int size, int x) {
    int i; // Declare loop variable outside the for loop
    for (i = 0; i < size; i++) {
        if (q[i] == x)
            return i;
    }
    return -1;
}

// Function to implement the Second Chance Page Replacement Algorithm
void Second_Chance_Replacement(int *t, int n, int capacity) {
    int *q = (int *)malloc(capacity * sizeof(int));  // Queue implementation using array
    bool *bitref = (bool *)malloc(capacity * sizeof(bool));  // Reference bits
    int qSize = 0;  // Current size of queue

    // Variables to track hits and faults
    int hits = 0, faults = 0;
    int ptr = 0;  // Pointer to track the position in the queue

    // Initialize the reference bits array
    int i; // Declare loop variable for initialization
    for (i = 0; i < capacity; i++) {
        bitref[i] = false;
    }

    for (i = 0; i < n; i++) {
        int pageIndex = findPage(q, qSize, t[i]);

        // If the page is already in memory (hit)
        if (pageIndex != -1) {
            hits++;
            bitref[pageIndex] = true;  // Set reference bit to 1 (second chance)
        }
        // If it's a fault (page not found in memory)
        else {
            faults++;

            // If there's still room in the queue
            if (qSize < capacity) {
                q[qSize] = t[i];  // Insert page into memory
                bitref[qSize] = false;  // Initially set reference bit to 0
                qSize++;
            }
            // If the queue is full, apply the Second Chance algorithm
            else {
                while (true) {
                    // Check the reference bit of the page at the current pointer position
                    if (bitref[ptr] == false) {
                        // If reference bit is 0, replace the page
                        q[ptr] = t[i];
                        bitref[ptr] = false; // Set reference bit for the replaced page
                        break;
                    } else {
                        // If reference bit is 1, reset it and move to the next page
                        bitref[ptr] = false;
                        ptr = (ptr + 1) % capacity;  // Circular increment
                    }
                }

                // Move pointer to the next position after replacement
                ptr = (ptr + 1) % capacity;
            }
        }
    }

    printf("Hits: %d\nFaults: %d\n", hits, faults);

    // Free dynamically allocated memory
    free(q);
    free(bitref);
}

// Driver code
int main() {
    int t[] = { 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 3};
    int n = sizeof(t) / sizeof(t[0]);
    int capacity = 3;
    Second_Chance_Replacement(t, n, capacity);

    return 0;
}

//10a

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILES 10
#define MAX_NAME_LENGTH 20
#define DISK_SIZE 100

// Structure to represent a file
struct File {
    char name[MAX_NAME_LENGTH];
    int size;
    int startBlock;
};

// Structure to represent the disk
struct Disk {
    char blocks[DISK_SIZE]; // '0' indicates free, '1' indicates allocated
    struct File files[MAX_FILES];
    int fileCount;
};

// Function to initialize the disk
void initializeDisk(struct Disk *disk) {
    for (int i = 0; i < DISK_SIZE; i++) {
        disk->blocks[i] = '0'; // Mark all blocks as free
    }
    disk->fileCount = 0;
}

// Function to allocate a file sequentially
int allocateFile(struct Disk *disk, const char *name, int size) {
    // Check if the file count exceeds the maximum allowed files
    if (disk->fileCount >= MAX_FILES) {
        printf("Error: Maximum number of files reached.\n");
        return -1;
    }

    // Find a contiguous space in the disk
    int start = -1, count = 0;
    for (int i = 0; i < DISK_SIZE; i++) {
        if (disk->blocks[i] == '0') { // Free block
            if (start == -1) {
                start = i; // Mark the start of a potential allocation
            }
            count++; // Increment count of contiguous free blocks
            if (count == size) { // Enough space found
                // Allocate the file
                struct File newFile;
                strcpy(newFile.name, name);
                newFile.size = size;
                newFile.startBlock = start;

                for (int j = start; j < start + size; j++) {
                    disk->blocks[j] = '1'; // Mark blocks as allocated
                }

                disk->files[disk->fileCount++] = newFile; // Add file to the disk
                printf("File '%s' allocated at block %d.\n", name, start);
                return 0; // Success
            }
        } else {
            // Reset if we hit an allocated block
            start = -1;
            count = 0;
        }
    }
    
    printf("Error: Not enough contiguous space to allocate file '%s'.\n", name);
    return -1; // Allocation failed
}

// Function to display the current state of the disk
void displayDiskState(struct Disk *disk) {
    printf("\nDisk State:\n");
    for (int i = 0; i < DISK_SIZE; i++) {
        printf("%c ", disk->blocks[i]);
    }
    printf("\n\nAllocated Files:\n");
    for (int i = 0; i < disk->fileCount; i++) {
        printf("File Name: %s, Size: %d, Starting Block: %d\n",
            disk->files[i].name, disk->files[i].size, disk->files[i].startBlock);
    }
}

int main() {
    struct Disk disk;
    initializeDisk(&disk);

    // Simulate file allocations
    allocateFile(&disk, "File1.txt", 10);
    allocateFile(&disk, "File2.txt", 20);
    allocateFile(&disk, "File3.txt", 5);
    allocateFile(&disk, "File4.txt", 15);
    allocateFile(&disk, "File5.txt", 30); // This should fail

    displayDiskState(&disk);

    return 0;
}


//11b

#include <stdio.h>
#include <stdlib.h>

void SSTF(int requests[], int n, int head) {
    int total_head_movement = 0;
    int completed[100] = {0}; // To track completed requests
    int current_index, i;

    // Display the initial head position
    printf("Initial Head Position: %d\n", head);
    printf("Disk Head Movement Sequence:\n");

    for (int count = 0; count < n; count++) {
        int min_distance = 10000; // Set to a large value
        int closest_index = -1;

        // Find the closest request
        for (i = 0; i < n; i++) {
            if (!completed[i]) { // Check if request is not completed
                int distance = abs(requests[i] - head);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_index = i; // Update closest request
                }
            }
        }

        // Move the head to the closest request
        if (closest_index != -1) {
            printf("%d -> %d\n", head, requests[closest_index]);
            total_head_movement += min_distance;
            head = requests[closest_index];
            completed[closest_index] = 1; // Mark as completed
        }
    }

    // Display the total head movement
    printf("Total Distance Covered in Tracks: %d\n", total_head_movement);
}

int main() {
    // Self-initialized values for disk requests and head position
    int requests[] = {95, 180, 34, 119, 11, 123, 62, 64};
    int n = sizeof(requests) / sizeof(requests[0]); // Number of disk requests
    int head = 50; // Initial position of the disk head

    // Run SSTF Disk Scheduling Algorithm
    SSTF(requests, n, head);

    return 0;
}


//11 a

#include <stdio.h>
#include <stdlib.h>

void FCFS(int requests[], int n, int head) {
    int total_head_movement = 0;
    int i;  // Declare the loop variable outside the for loop

    // Display the initial head position
    printf("Initial Head Position: %d\n", head);
    printf("Disk Head Movement Sequence:\n");

    // Service each request in the order they appear
    for (i = 0; i < n; i++) {
        // Display the movement from the current head to the next request
        printf("%d -> %d\n", head, requests[i]);

        // Calculate the head movement for the current request
        int movement = abs(requests[i] - head);
        total_head_movement += movement;

        // Move the head to the current request's position
        head = requests[i];
    }

    // Display the total head movement
    printf("Total Distance Covered in Tracks: %d\n", total_head_movement);
}

int main() {
    // Self-initialized values for disk requests and head position
    int requests[] = {95, 180, 34, 119, 11, 123, 62, 64};
    int n = sizeof(requests) / sizeof(requests[0]); // Number of disk requests
    int head = 50; // Initial position of the disk head

    // Run FCFS Disk Scheduling Algorithm
    FCFS(requests, n, head);

    return 0;
}



//1o b

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILES 10
#define DISK_SIZE 20
#define MAX_NAME_LENGTH 20

// Structure to represent a block in the disk
struct Block {
    int nextBlock; // Index of the next block; -1 if no next block
    char data[50]; // Data stored in the block (can be the file name)
};

// Structure to represent a file
struct File {
    char name[MAX_NAME_LENGTH];
    int startBlock; // Starting block index of the file
    int size; // Size of the file in blocks
};

// Structure to represent the disk
struct Disk {
    struct Block blocks[DISK_SIZE]; // Array of blocks
    struct File files[MAX_FILES]; // Array of files
    int fileCount; // Number of files currently allocated
};

// Function to initialize the disk
void initializeDisk(struct Disk *disk) {
    for (int i = 0; i < DISK_SIZE; i++) {
        disk->blocks[i].nextBlock = -1; // Initialize nextBlock as -1 (no next block)
        strcpy(disk->blocks[i].data, ""); // Initialize data as empty
    }
    disk->fileCount = 0;
}

// Function to allocate a file using linked allocation
int allocateFile(struct Disk *disk, const char *name, int size) {
    if (disk->fileCount >= MAX_FILES) {
        printf("Error: Maximum number of files reached.\n");
        return -1;
    }

    int previousBlock = -1; // To keep track of the previous block
    int firstBlock = -1; // To keep track of the first block

    // Find free blocks for allocation
    for (int i = 0; i < DISK_SIZE; i++) {
        if (disk->blocks[i].nextBlock == -1 && strcmp(disk->blocks[i].data, "") == 0) {
            // Allocate the block
            if (firstBlock == -1) {
                firstBlock = i; // Mark the first block
            } else {
                disk->blocks[previousBlock].nextBlock = i; // Link the previous block to the current block
            }
            previousBlock = i; // Update the previous block index

            // Store the file name in the block data
            snprintf(disk->blocks[i].data, sizeof(disk->blocks[i].data), "%s", name);
            size--; // Decrease the size of the file being allocated

            if (size == 0) { // If the file is fully allocated
                break;
            }
        }
    }

    // Check if the file was fully allocated
    if (size > 0) {
        printf("Error: Not enough space to allocate file '%s'.\n", name);
        return -1; // Allocation failed
    }

    // Set the nextBlock of the last allocated block to -1
    if (previousBlock != -1) {
        disk->blocks[previousBlock].nextBlock = -1; // End of the linked list
    }

    // Store the file information
    struct File newFile;
    strcpy(newFile.name, name);
    newFile.startBlock = firstBlock;
    newFile.size = previousBlock - firstBlock + 1; // Calculate the size of the file in blocks

    disk->files[disk->fileCount++] = newFile; // Add file to the disk
    printf("File '%s' allocated starting at block %d.\n", name, firstBlock);
    return 0; // Success
}

// Function to display the current state of the disk
void displayDiskState(struct Disk *disk) {
    printf("\nDisk State:\n");
    for (int i = 0; i < DISK_SIZE; i++) {
        printf("[Block %2d]: %s (Next: %d)  ", i, disk->blocks[i].data, disk->blocks[i].nextBlock);
    }
    printf("\n\nAllocated Files:\n");
    for (int i = 0; i < disk->fileCount; i++) {
        printf("File Name: %s, Starting Block: %d, Size: %d\n",
            disk->files[i].name, disk->files[i].startBlock, disk->files[i].size);
    }
}

int main() {
    struct Disk disk;
    initializeDisk(&disk);

    // Simulate file allocations
    allocateFile(&disk, "File1.txt", 3);
    allocateFile(&disk, "File2.txt", 4);
    allocateFile(&disk, "File3.txt", 2);
    allocateFile(&disk, "File4.txt", 5);
    allocateFile(&disk, "File5.txt", 3); // This should succeed

    displayDiskState(&disk);

    return 0;
}


//11c

#include <stdio.h>
#include <stdlib.h>

void SCAN(int requests[], int n, int head, int direction) {
    int total_head_movement = 0;
    int current_index, i;

    // Display the initial head position
    printf("Initial Head Position: %d\n", head);
    printf("Disk Head Movement Sequence:\n");

    // Sort the requests
    for (i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (requests[i] > requests[j]) {
                int temp = requests[i];
                requests[i] = requests[j];
                requests[j] = temp;
            }
        }
    }

    // Determine the position of the head in the sorted array
    for (i = 0; i < n; i++) {
        if (requests[i] >= head) {
            current_index = i;
            break;
        }
    }

    // Move in the specified direction
    if (direction == 1) { // Moving right
        // Service requests to the right of the head
        for (i = current_index; i < n; i++) {
            printf("%d -> %d\n", head, requests[i]);
            total_head_movement += abs(requests[i] - head);
            head = requests[i];
        }
        // Move to the end of the disk and then service requests to the left
        printf("%d -> %d\n", head, 199); // Assuming the disk size is 200
        total_head_movement += abs(199 - head);
        head = 199;
        
        for (i = n - 1; i >= 0; i--) {
            printf("%d -> %d\n", head, requests[i]);
            total_head_movement += abs(requests[i] - head);
            head = requests[i];
        }
    } else { // Moving left
        // Service requests to the left of the head
        for (i = current_index - 1; i >= 0; i--) {
            printf("%d -> %d\n", head, requests[i]);
            total_head_movement += abs(requests[i] - head);
            head = requests[i];
        }
        // Move to the start of the disk and then service requests to the right
        printf("%d -> %d\n", head, 0);
        total_head_movement += abs(0 - head);
        head = 0;

        for (i = 0; i < n; i++) {
            printf("%d -> %d\n", head, requests[i]);
            total_head_movement += abs(requests[i] - head);
            head = requests[i];
        }
    }

    // Display the total head movement
    printf("Total Distance Covered in Tracks: %d\n", total_head_movement);
}

int main() {
    // Self-initialized values for disk requests and head position
    int requests[] = {95, 180, 34, 119, 11, 123, 62, 64};
    int n = sizeof(requests) / sizeof(requests[0]); // Number of disk requests
    int head = 50; // Initial position of the disk head
    int direction = 1; // 1 for right, 0 for left

    // Run SCAN Disk Scheduling Algorithm
    SCAN(requests, n, head, direction);

    return 0;
}


//11d

#include <stdio.h>
#include <stdlib.h>

void C_SCAN_Top_Bottom(int requests[], int n, int head, int disk_size)
{
    int total_head_movement = 0;
    int sorted_requests[n + 2]; // Include 2 more for the boundary points (0 and disk_size - 1)
    int i, j;

    // Copy the requests and add the boundary points (0 and disk_size - 1)
    for (i = 0; i < n; i++)
        sorted_requests[i] = requests[i];

    sorted_requests[n] = 0;                 // Add the lowest track (0)
    sorted_requests[n + 1] = disk_size - 1; // Add the highest track (disk_size - 1)

    // Sort the request array along with boundary points
    for (i = 0; i < n + 2; i++)
    {
        for (j = i + 1; j < n + 2; j++)
        {
            if (sorted_requests[i] > sorted_requests[j])
            {
                int temp = sorted_requests[i];
                sorted_requests[i] = sorted_requests[j];
                sorted_requests[j] = temp;
            }
        }
    }

    // Display the initial head position
    printf("Initial Head Position: %d\n", head);
    printf("Disk Head Movement Sequence (Top-Bottom C-SCAN Approach):\n");

    // Find the position of the head in the sorted array
    int start_index = 0;
    for (i = 0; i < n + 2; i++)
    {
        if (sorted_requests[i] >= head)
        {
            start_index = i;
            break;
        }
    }

    // Move Upwards (Top-Down)
    for (i = start_index; i < n + 2; i++)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // Jump directly from the highest track to the lowest track (Circular behavior)
    if (head != 0)
    {
        printf("%d -> 0 (Jump)\n", head);
        total_head_movement += abs(head - 0);
        head = 0;
    }

    // Move Upwards again from the lowest track
    for (i = 0; i < start_index; i++)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // Display the total head movement
    printf("Total Distance Covered in Tracks (Top-Bottom C-SCAN): %d\n", total_head_movement);
}

void C_SCAN_Bottom_Top(int requests[], int n, int head, int disk_size)
{
    int total_head_movement = 0;
    int sorted_requests[n + 2]; // Include 2 more for the boundary points (0 and disk_size - 1)
    int i, j;

    // Copy the requests and add the boundary points (0 and disk_size - 1)
    for (i = 0; i < n; i++)
        sorted_requests[i] = requests[i];

    sorted_requests[n] = 0;                 // Add the lowest track (0)
    sorted_requests[n + 1] = disk_size - 1; // Add the highest track (disk_size - 1)

    // Sort the request array along with boundary points
    for (i = 0; i < n + 2; i++)
    {
        for (j = i + 1; j < n + 2; j++)
        {
            if (sorted_requests[i] > sorted_requests[j])
            {
                int temp = sorted_requests[i];
                sorted_requests[i] = sorted_requests[j];
                sorted_requests[j] = temp;
            }
        }
    }

    // Display the initial head position
    printf("\nInitial Head Position: %d\n", head);
    printf("Disk Head Movement Sequence (Bottom-Top C-SCAN Approach):\n");

    // Find the position of the head in the sorted array
    int start_index = 0;
    for (i = 0; i < n + 2; i++)
    {
        if (sorted_requests[i] >= head)
        {
            start_index = i;
            break;
        }
    }

    // Move Downwards (Bottom-Up)
    for (i = start_index - 1; i >= 0; i--)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // Jump directly from the lowest track to the highest track (Circular behavior)
    if (head != disk_size - 1)
    {
        printf("%d -> %d (Jump)\n", head, disk_size - 1);
        total_head_movement += abs(head - (disk_size - 1));
        head = disk_size - 1;
    }

    // Move Downwards again from the highest track
    for (i = n + 1; i >= start_index; i--)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // Display the total head movement
    printf("Total Distance Covered in Tracks (Bottom-Top C-SCAN): %d\n", total_head_movement);
}

int main()
{
    // Self-initialized values for disk requests and head position
    int requests[] = {95, 180, 34, 119, 11, 123, 62, 64}; // Disk request queue
    int n = sizeof(requests) / sizeof(requests[0]);       // Number of disk requests
    int head = 50;
    // Initial position of the disk hea
    int disk_size = 200; // Size of the disk (number of tracks)

    // Run C-SCAN Disk Scheduling Algorithm (Top-Bottom approach)
    C_SCAN_Top_Bottom(requests, n, head, disk_size);

    // Run C-SCAN Disk Scheduling Algorithm (Bottom-Top approach)
    C_SCAN_Bottom_Top(requests, n, head, disk_size);

    return 0;
}


//11e

#include <stdio.h>
#include <stdlib.h>

void LOOK_Top_Bottom(int requests[], int n, int head)
{
    int total_head_movement = 0;
    int sorted_requests[n]; // Array to store sorted requests
    int i, j;

    // Copy the requests
    for (i = 0; i < n; i++)
        sorted_requests[i] = requests[i];

    // Sort the request array
    for (i = 0; i < n; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            if (sorted_requests[i] > sorted_requests[j])
            {
                int temp = sorted_requests[i];
                sorted_requests[i] = sorted_requests[j];
                sorted_requests[j] = temp;
            }
        }
    }

    // Display the initial head position
    printf("Initial Head Position: %d\n", head);
    printf("Disk Head Movement Sequence (Top-Bottom LOOK Approach):\n");

    // Find the position of the head in the sorted array
    int start_index = 0;
    for (i = 0; i < n; i++)
    {
        if (sorted_requests[i] >= head)
        {
            start_index = i;
            break;
        }
    }

    // Move Upwards (Top-Down)
    for (i = start_index; i < n; i++)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // After reaching the highest request, reverse direction
    for (i = start_index - 1; i >= 0; i--)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // Display the total head movement
    printf("Total Distance Covered in Tracks (Top-Bottom LOOK): %d\n", total_head_movement);
}

void LOOK_Bottom_Top(int requests[], int n, int head)
{
    int total_head_movement = 0;
    int sorted_requests[n]; // Array to store sorted requests
    int i, j;

    // Copy the requests
    for (i = 0; i < n; i++)
        sorted_requests[i] = requests[i];

    // Sort the request array
    for (i = 0; i < n; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            if (sorted_requests[i] > sorted_requests[j])
            {
                int temp = sorted_requests[i];
                sorted_requests[i] = sorted_requests[j];
                sorted_requests[j] = temp;
            }
        }
    }

    // Display the initial head position
    printf("\nInitial Head Position: %d\n", head);
    printf("Disk Head Movement Sequence (Bottom-Top LOOK Approach):\n");

    // Find the position of the head in the sorted array
    int start_index = 0;
    for (i = 0; i < n; i++)
    {
        if (sorted_requests[i] >= head)
        {
            start_index = i;
            break;
        }
    }

    // Move Downwards (Bottom-Up)
    for (i = start_index - 1; i >= 0; i--)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // After reaching the lowest request, reverse direction
    for (i = start_index; i < n; i++)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // Display the total head movement
    printf("Total Distance Covered in Tracks (Bottom-Top LOOK): %d\n", total_head_movement);
}

int main()
{
    // Self-initialized values for disk requests and head position
    int requests[] = {95, 180, 34, 119, 11, 123, 62, 64}; // Disk request queue
    int n = sizeof(requests) / sizeof(requests[0]);       // Number of disk requests
    int head = 50;

    // Run LOOK Disk Scheduling Algorithm (Top-Bottom approach)
    LOOK_Top_Bottom(requests, n, head);

    // Run LOOK Disk Scheduling Algorithm (Bottom-Top approach)
    LOOK_Bottom_Top(requests, n, head);

    return 0;
}
 

//11f

#include <stdio.h>
#include <stdlib.h>

void C_LOOK_Top_Bottom(int requests[], int n, int head)
{
    int total_head_movement = 0;
    int sorted_requests[n]; // Array to store sorted requests
    int i, j;

    // Copy the requests
    for (i = 0; i < n; i++)
        sorted_requests[i] = requests[i];

    // Sort the request array
    for (i = 0; i < n; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            if (sorted_requests[i] > sorted_requests[j])
            {
                int temp = sorted_requests[i];
                sorted_requests[i] = sorted_requests[j];
                sorted_requests[j] = temp;
            }
        }
    }

    // Display the initial head position
    printf("Initial Head Position: %d\n", head);
    printf("Disk Head Movement Sequence (Top-Bottom C-LOOK Approach):\n");

    // Find the position of the head in the sorted array
    int start_index = 0;
    for (i = 0; i < n; i++)
    {
        if (sorted_requests[i] >= head)
        {
            start_index = i;
            break;
        }
    }

    // Move Upwards (Top-Down)
    for (i = start_index; i < n; i++)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // Jump directly to the lowest unserviced request
    if (start_index > 0)
    {
        printf("%d -> %d (Jump)\n", head, sorted_requests[0]);
        total_head_movement += abs(head - sorted_requests[0]);
        head = sorted_requests[0];

        // Move Upwards again from the lowest request
        for (i = 0; i < start_index; i++)
        {
            printf("%d -> %d\n", head, sorted_requests[i]);
            total_head_movement += abs(sorted_requests[i] - head);
            head = sorted_requests[i];
        }
    }

    // Display the total head movement
    printf("Total Distance Covered in Tracks (Top-Bottom C-LOOK): %d\n", total_head_movement);
}

void C_LOOK_Bottom_Top(int requests[], int n, int head)
{
    int total_head_movement = 0;
    int sorted_requests[n]; // Array to store sorted requests
    int i, j;

    // Copy the requests
    for (i = 0; i < n; i++)
        sorted_requests[i] = requests[i];

    // Sort the request array
    for (i = 0; i < n; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            if (sorted_requests[i] > sorted_requests[j])
            {
                int temp = sorted_requests[i];
                sorted_requests[i] = sorted_requests[j];
                sorted_requests[j] = temp;
            }
        }
    }

    // Display the initial head position
    printf("\nInitial Head Position: %d\n", head);
    printf("Disk Head Movement Sequence (Bottom-Top C-LOOK Approach):\n");

    // Find the position of the head in the sorted array
    int start_index = 0;
    for (i = 0; i < n; i++)
    {
        if (sorted_requests[i] >= head)
        {
            start_index = i;
            break;
        }
    }

    // Move Downwards (Bottom-Up)
    for (i = start_index - 1; i >= 0; i--)
    {
        printf("%d -> %d\n", head, sorted_requests[i]);
        total_head_movement += abs(sorted_requests[i] - head);
        head = sorted_requests[i];
    }

    // Jump directly to the highest unserviced request
    if (start_index < n)
    {
        printf("%d -> %d (Jump)\n", head, sorted_requests[n - 1]);
        total_head_movement += abs(head - sorted_requests[n - 1]);
        head = sorted_requests[n - 1];

        // Move Downwards again from the highest request
        for (i = n - 2; i >= start_index; i--)
        {
            printf("%d -> %d\n", head, sorted_requests[i]);
            total_head_movement += abs(sorted_requests[i] - head);
            head = sorted_requests[i];
        }
    }

    // Display the total head movement
    printf("Total Distance Covered in Tracks (Bottom-Top C-LOOK): %d\n", total_head_movement);
}

int main()
{
    // Self-initialized values for disk requests and head position
    int requests[] = {95, 180, 34, 119, 11, 123, 62, 64}; // Disk request queue
    int n = sizeof(requests) / sizeof(requests[0]);       // Number of disk requests
    int head = 50;

    // Run C-LOOK Disk Scheduling Algorithm (Top-Bottom approach)
    C_LOOK_Top_Bottom(requests, n, head);

    // Run C-LOOK Disk Scheduling Algorithm (Bottom-Top approach)
    C_LOOK_Bottom_Top(requests, n, head);

    return 0;
}
*/
