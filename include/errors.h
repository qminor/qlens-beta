// ERRORS.H: Contains functions for error checking

#ifndef ERRORS_H
#define ERRORS_H

void die(void);
void die(char *, ...);
void warn(char *, ...);
void warn(bool, char *, ...);

void openerror(char *);
void readerror(char *);
void writeerror(char *);
void warn_openerror(char *);
void warn_readerror(char *);
void warn_writeerror(char *);

#endif // ERRORS_H
