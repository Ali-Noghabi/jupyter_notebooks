#include "Mutex.h"

Mutex::Mutex() : lock(false) {}

void Mutex::acquire() {
    bool expected = false;
    while (!lock.compare_exchange_strong(expected, true)) {
        expected = false;  // Reset expected value for the next iteration
    }
}

void Mutex::release() {
    lock.store(false);
}
