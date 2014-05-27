#ifndef TIMER_HH
#define TIMER_HH

#include <iostream>
#include <chrono>
#include <string>


class StartTimer
{
private:
    const std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> begin;
    double time_seconds;

public:
    StartTimer(const std::string &name)
        :   name (name)
        ,   begin (std::chrono::high_resolution_clock::now())
    {}

    void end()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto interval = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
        time_seconds = (interval.count()*1e-6);
        std::cout << name << ": " << time_seconds << " s" << std::endl;
    }

    double seconds()
    {
        return time_seconds;
    }
};

#endif // TIMER_HH
