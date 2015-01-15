all::	thom

thom:	thom.c
	gcc -o thom thom.c -Wall -O2 -fomit-frame-pointer -lm -msse2

clean:
	rm -f *~ thom
