// snow_math.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "SimulationDriver.h"
#include "particle.h"

#define Model 0

std::vector<std::string> split_string(std::string s, char c)
{
    std::vector<std::string> elements;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, c))
    {
        if (item != "")
            elements.push_back(item);
    }
    return elements;
}

template <class T>
T stringToNum(const std::string& str)
{
    std::istringstream iss(str);
    T num;
    iss >> num;
    return num;
}

int main()
{
    using FV = Eigen::Matrix<float, 3, 1>;
    SimulationDriver driver;

#if Model == 0  // Cube
    int n = 30;
    float l = 0.98f * (float)n / 10.f;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                Particle p;
                p.position = FV(i * l / (float)n,
                                j * l / (float)n,
                                k * l / (float)n);
                p.position += FV(1.05f, 0.05f, 1.05f);
                driver.ss.particles.push_back(p);
            }
        }
    }
#else   // Bunny
    std::string pointFile = "data/points.txt";
    std::ifstream infile1(pointFile);
    assert(infile1.is_open());
    std::string aLine;
    bool isFirstLine = true;

    int pointNum = 0;
    int dim = 0;

    while (std::getline(infile1, aLine))
    {
        std::vector<std::string> elems = split_string(aLine, ' ');
        if (isFirstLine)
        {
            pointNum = stringToNum<int>(elems[0]);
            dim = stringToNum<int>(elems[1]);
            isFirstLine = false;
            std::cout << pointNum << std::endl;
        }
        else
        {
            FV xTmp = FV::Zero();
            xTmp[0] = stringToNum<float>(elems[0]) * 50.f;
            xTmp[1] = stringToNum<float>(elems[1]) * 50.f;
            xTmp[2] = stringToNum<float>(elems[2]) * 50.f;
            Particle p;
            p.position = xTmp;
            p.position += FV(5.f, 0.05f, 5.f);
            driver.ss.particles.push_back(p);
        }
    }
#endif 

    driver.run();

    std::cout << "Hello World!\n";
}
