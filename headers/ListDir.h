#ifndef LISTDIR_H
#define LISTDIR_H

#pragma once
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
using namespace std;

vector<string> ListDirectories(string dirName) {
    DIR *dir;
    struct dirent *ent;
    vector<string> result;
    dir = opendir(dirName.c_str());
    dirName+="/";
    if (dir != NULL) {

        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] >= '0')
                result.push_back(dirName+(string)ent->d_name);
                //printf("%s\n", ent->d_name);
        }
        closedir(dir);
        sort(result.begin(),result.end());
        return result;
    } else {
        /* could not open directory */
        perror("no valid dir");
        return result;
    }
}

#endif // LISTDIR_H
