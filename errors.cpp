#include "errors.h"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>

void die(const std::string sstring, ...)
{
	char s[sstring.length() + 1];
	std::strcpy(s, sstring.c_str());
	va_list ap;
	char *p, *sval;
	fprintf(stderr, "Error: ");

	va_start(ap, sstring);
	for (p = s; *p; p++) {
		if (*p != '%') {
			putc(*p, stderr);
			continue;
		}
		if (*++p=='s')
			for (sval = va_arg(ap, char *); *sval; sval++)
				putc(*sval, stderr);
		else if ((*p=='g') || (*p=='f'))
			fprintf(stderr, "%g", va_arg(ap, double));
		else if (*p=='i')
			fprintf(stderr, "%i", va_arg(ap, int));
		else if (*p=='c')
			fprintf(stderr, "%c", va_arg(ap, int));
		else putc(*p, stderr);
	}
	fprintf(stderr, "\n");
	va_end(ap);
	//getc(stdin);
	exit(1);
}

void die(void) { exit(1); }

void warn(const std::string sstring, ...)
{
	char s[sstring.length() + 1];
	std::strcpy(s, sstring.c_str());
	va_list ap;
	char *p, *sval;
	fprintf(stderr, "*WARNING*: ");

	va_start(ap, sstring);
	for (p = s; *p; p++) {
		if (*p != '%') {
			putc(*p, stderr);
			continue;
		}
		if (*++p=='s')
			for (sval = va_arg(ap, char *); *sval; sval++)
				putc(*sval, stderr);
		else if ((*p=='g') || (*p=='f'))
			fprintf(stderr, "%g", va_arg(ap, double));
		else if (*p=='c')
			fprintf(stderr, "%c", va_arg(ap, int));
		else if (*p=='i')
			fprintf(stderr, "%i", va_arg(ap, int));
		else putc(*p, stderr);
	}
	fprintf(stderr, "\n");

	return;
}

void warn(const bool warnings_on, const std::string sstring, ...)
{
	if (warnings_on==false) return;
	char s[sstring.length() + 1];
	std::strcpy(s, sstring.c_str());

	va_list ap;
	char *p, *sval;
	fprintf(stderr, "*WARNING*: ");

	va_start(ap, sstring);
	for (p = s; *p; p++) {
		if (*p != '%') {
			putc(*p, stderr);
			continue;
		}
		if (*++p=='s')
			for (sval = va_arg(ap, char *); *sval; sval++)
				putc(*sval, stderr);
		else if ((*p=='g') || (*p=='f'))
			fprintf(stderr, "%g", va_arg(ap, double));
		else if (*p=='c')
			fprintf(stderr, "%c", va_arg(ap, int));
		else if (*p=='i')
			fprintf(stderr, "%i", va_arg(ap, int));
		else putc(*p, stderr);
	}
	fprintf(stderr, "\n");

	return;
}

void openerror(char *errfile) { fprintf(stderr,"could not open file %s\n", errfile); exit(1); }
void readerror(char *errfile) { fprintf(stderr,"could not read arguments from file %s\n", errfile); exit(1); }
void writeerror(char *errfile) { fprintf(stderr,"could not write to file %s\n", errfile); exit(1); }
void warn_openerror(char *errfile) { fprintf(stderr,"Error: could not open file %s\n", errfile); }
void warn_readerror(char *errfile) { fprintf(stderr,"Error: could not read arguments from file %s\n", errfile); }
void warn_writeerror(char *errfile) { fprintf(stderr,"Error: could not write to file %s\n", errfile); }

