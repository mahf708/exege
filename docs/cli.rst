Command line
============

Every command is a client of the Python API: anything it does, a notebook can. ``--json``,
where a command has it, writes the settings, provenance and results of what it printed.

.. click:: exege._cli:cli
   :prog: exege
   :nested: full
