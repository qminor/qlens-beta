#include "errors.h"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

void die(char *s, ...)
{
	va_list ap;
	char *p, *sval;
	fprintf(stderr, "Error: ");

	va_start(ap, s);
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
	getc(stdin);
	exit(1);
}

void die(void) { exit(1); }

void warn(char *s, ...)
{
	va_list ap;
	char *p, *sval;
	fprintf(stderr, "*WARNING*: ");

	va_start(ap, s);
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

void warn(bool warnings_on, char *s, ...)
{
	if (warnings_on==false) return;

	va_list ap;
	char *p, *sval;
	fprintf(stderr, "*WARNING*: ");

	va_start(ap, s);
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

