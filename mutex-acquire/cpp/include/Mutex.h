#ifndef MUTEX_H
#define MUTEX_H

#include <atomic>

class Mutex {
public:
    Mutex();
    void acquire();
    void release();
private:
    std::atomic<bool> lock;
};

#endif
