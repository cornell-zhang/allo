/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//---------------------------------------------------------
// Timer.h
//---------------------------------------------------------
#ifndef __TIMER_H__
#define __TIMER_H__
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>

#define TIMER_ON

//---------------------------------------------------------
// Timer is an object which helps profile programs using
// the clock() function.
// - By default, a timer is stopped when you instantiate it
//   and must be started manually
// - Passing True to the constructor starts the timer when
//   it is constructed
// - When the timer is destructed it prints stats to stdout
//---------------------------------------------------------
class Timer {

  #ifdef TIMER_ON

    char binName[50];
    unsigned nCalls;
    timeval ts_start;
    float totalTime;
    
    public:
      //------------------------------------------------------------------
      // constructor
      //------------------------------------------------------------------
      Timer (const char* Name="", bool On=false) {
        if (On) {
          // record the start time
          gettimeofday(&ts_start, NULL);
          nCalls = 1;
        }
        else {
          nCalls = 0;
        }
        totalTime = 0;	
        strcpy(binName, Name);
      }

      //------------------------------------------------------------------
      // destructor
      //------------------------------------------------------------------
      ~Timer () {
        // on being destroyed, print the average and total time
        if (nCalls > 0) {
          printf ("%-20s: ", binName);
          printf ("%6d calls; ", nCalls);
          printf ("%7.3f msecs total time\n", 1000*totalTime);
          //printf ("%7.4f msecs average time;\n", 1000*totalTime/nCalls);
        }
      }
      
      //------------------------------------------------------------------
      // start timer
      //------------------------------------------------------------------
      void start() {
        // record start time
        gettimeofday(&ts_start, NULL);
        nCalls++;
      }
      
      //------------------------------------------------------------------
      // stop timer
      //------------------------------------------------------------------
      void stop() {
        // get current time, add elapsed time to totalTime
        timeval ts_curr;
        gettimeofday(&ts_curr, NULL);
        totalTime += float(ts_curr.tv_sec - ts_start.tv_sec) +
                     float(ts_curr.tv_usec)*1e-6 - float(ts_start.tv_usec)*1e-6;
      }

  #else

    //--------------------------------------------------------------------
    // all methods do nothing if TIMER_ON is not set
    //--------------------------------------------------------------------
    public:
      Timer (const char* Name, bool On=true) {}
      void start() {}
      void stop() {}

  #endif
};

#endif
