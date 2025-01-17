Internet topology network, part of the Koblenz Network Collection
===========================================================================

This directory contains the TSV and related files of the topology network:

This is the network of connections between autonomous systems of the Internet. The nodes are autonomous systems (AS), i.e. collections of connected IP routing prefixes controlled by independent network operators. Edges are connections between autonomous systems. Multiple edges may connect two nodes, each representing an individual connection in time. Edges are annotated with the timepoint of the connection.


More information about the network is provided here:
http://konect.uni-koblenz.de/networks/topology

Files:
    meta.topology -- Metadata about the network
    out.topology -- The adjacency matrix of the network in space separated values format, with one edge per line
      The meaning of the columns in out.topology are:
        First column: ID of from node
        Second column: ID of to node
        Third column: edge weight
        Fourth column: timestamp of the edge


Complete documentation about the file format can be found in the KONECT
handbook, in the section File Formats, available at:

http://konect.uni-koblenz.de/publications

All files are licensed under a Creative Commons Attribution-ShareAlike 2.0 Germany License.
For more information concerning license visit http://konect.uni-koblenz.de/license.



Use the following References for citation:

@MISC{konect:2016:topology,
    title = {Internet topology network dataset -- {KONECT}},
    month = sep,
    year = {2016},
    url = {http://konect.uni-koblenz.de/networks/topology}
}

@article{konect:zhang05,
        title = {Collecting the {Internet} {AS}-level Topology},
        author = {Zhang, Beichuan and Liu, Raymond and Massey, Daniel
                  and Zhang, Lixia},
        journal = {SIGCOMM Computer Communication Review},
        volume = {35},
        number = {1},
        year = {2005},
        pages = {53--61},
}


@inproceedings{konect,
	title = {{KONECT} -- {The} {Koblenz} {Network} {Collection}},
	author = {Jérôme Kunegis},
	year = {2013},
	booktitle = {Proc. Int. Conf. on World Wide Web Companion},
	pages = {1343--1350},
	url = {http://userpages.uni-koblenz.de/~kunegis/paper/kunegis-koblenz-network-collection.pdf},
	url_presentation = {http://userpages.uni-koblenz.de/~kunegis/paper/kunegis-koblenz-network-collection.presentation.pdf},
}
