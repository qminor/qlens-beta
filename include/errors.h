// ERRORS.H: Contains functions for error checking

#ifndef ERRORS_H
#define ERRORS_H

#include <string>

void die(void);
void die(const std::string, ...);
void warn(const std::string, ...);
void warn(const bool, const std::string, ...);

void openerror(char *);
void readerror(char *);
void writeerror(char *);
void warn_openerror(char *);
void warn_readerror(char *);
void warn_writeerror(char *);

#endif // ERRORS_H
