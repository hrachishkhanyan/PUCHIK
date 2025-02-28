#include <iostream>
#include <vector>
#include <fstream>
#include <string.h>
#include <string>

using namespace std;

string split(string s, char delim) {
    
}

int main(int argc, char const *argv[])
{
    fstream file;
    file.open("C:\\Users\\hrach\\PycharmProjects\\md_grid_project\\cpp\\test.txt", ios::in);

    
    if (file.is_open()) {
        string line;

        while (getline(file, line)) {
            char str[80];
            strcpy(str, line.c_str());
            strtok(str, " ");
            
            cout << str  << "\n";
        }
    }
    file.close();
    return 0;
}
