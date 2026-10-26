/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <stdlib.h>
#include "tagCircle21h7.h"

static uint64_t codedata[38] = {
   0x0000000000157863UL,
   0x0000000000047e28UL,
   0x00000000001383edUL,
   0x000000000000953cUL,
   0x00000000000da68bUL,
   0x00000000001cac50UL,
   0x00000000000bb215UL,
   0x000000000016ceeeUL,
   0x000000000005d4b3UL,
   0x00000000001ff751UL,
   0x00000000000efd16UL,
   0x0000000000072b3eUL,
   0x0000000000163103UL,
   0x0000000000106e56UL,
   0x00000000001996b9UL,
   0x00000000000c0234UL,
   0x00000000000624d2UL,
   0x00000000001fa985UL,
   0x00000000000344a5UL,
   0x00000000000762fbUL,
   0x000000000019e92bUL,
   0x0000000000043755UL,
   0x000000000001a4f4UL,
   0x000000000010fad8UL,
   0x0000000000001b52UL,
   0x000000000017e59fUL,
   0x00000000000e6f70UL,
   0x00000000000ed47aUL,
   0x00000000000c9931UL,
   0x0000000000014df2UL,
   0x00000000000a06f1UL,
   0x00000000000e5041UL,
   0x000000000012ec03UL,
   0x000000000016724eUL,
   0x00000000000af1a5UL,
   0x000000000008a8acUL,
   0x0000000000015b39UL,
   0x00000000001ec1e3UL,
};
apriltag_family_t *tagCircle21h7_create()
{
   apriltag_family_t *tf = calloc(1, sizeof(apriltag_family_t));
   tf->name = strdup("tagCircle21h7");
   tf->h = 7;
   tf->ncodes = 38;
   tf->codes = codedata;
   tf->nbits = 21;
   tf->bit_x = calloc(21, sizeof(uint32_t));
   tf->bit_y = calloc(21, sizeof(uint32_t));
   tf->bit_x[0] = 1;
   tf->bit_y[0] = -2;
   tf->bit_x[1] = 2;
   tf->bit_y[1] = -2;
   tf->bit_x[2] = 3;
   tf->bit_y[2] = -2;
   tf->bit_x[3] = 1;
   tf->bit_y[3] = 1;
   tf->bit_x[4] = 2;
   tf->bit_y[4] = 1;
   tf->bit_x[5] = 6;
   tf->bit_y[5] = 1;
   tf->bit_x[6] = 6;
   tf->bit_y[6] = 2;
   tf->bit_x[7] = 6;
   tf->bit_y[7] = 3;
   tf->bit_x[8] = 3;
   tf->bit_y[8] = 1;
   tf->bit_x[9] = 3;
   tf->bit_y[9] = 2;
   tf->bit_x[10] = 3;
   tf->bit_y[10] = 6;
   tf->bit_x[11] = 2;
   tf->bit_y[11] = 6;
   tf->bit_x[12] = 1;
   tf->bit_y[12] = 6;
   tf->bit_x[13] = 3;
   tf->bit_y[13] = 3;
   tf->bit_x[14] = 2;
   tf->bit_y[14] = 3;
   tf->bit_x[15] = -2;
   tf->bit_y[15] = 3;
   tf->bit_x[16] = -2;
   tf->bit_y[16] = 2;
   tf->bit_x[17] = -2;
   tf->bit_y[17] = 1;
   tf->bit_x[18] = 1;
   tf->bit_y[18] = 3;
   tf->bit_x[19] = 1;
   tf->bit_y[19] = 2;
   tf->bit_x[20] = 2;
   tf->bit_y[20] = 2;
   tf->width_at_border = 5;
   tf->total_width = 9;
   tf->reversed_border = true;
   return tf;
}

void tagCircle21h7_destroy(apriltag_family_t *tf)
{
   free(tf->bit_x);
   free(tf->bit_y);
   free(tf->name);
   free(tf);
}
