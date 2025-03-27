# How do we construct a dynamic and useful dependency graph?

## What should the dependency graph display?
 - Display courses and certificates for **one** category - we can expand later as needed
 - Only display courses and certificates that are relevant to the user's query - not the entire graph
 - Note: this is recomputed **every single query**

## Assume we are given (by chat or whatever)
 - General category that the user is interested in
 - Textual description of what job role(s) the user is interested in - this can come from user experience, what they're looking for, etc.
 - How much information the user is looking for - do they want a big picture, more granular with just the first few steps, etc.
 - Starting node or equivalent (user experience corresponding to level, category, starting job role, whether they want a course or a certificate, etc.) - this is **highly** specific

## Controlling graph width
Since our idea of what job role the user is looking for comes from text, we can compare this to the description of each job role using cosine similarity (or similar) to find the best role.

Job role descriptions can be found [here](https://certification.adobe.com/certifications/learn-more?tab=learnmore3).

We can simply set a threshold and include all nodes with a job role exceeding this threshold in the dependency graph

## Controlling graph depth
This is a little more interesting. Do we define a simple discrete level-based system where, depending on how much the user wants to know, we display $n$ rows of the graph? Or, do we take into account the different metadatas of each node (time taken to complete the course, cost of a certificate, etc.)? These are valid things that the user may specify and request a trajectory for, so if this is the case, the graph needs to properly represent that. 

For now, let's operate off a discrete level-based system because introducing specific constraints that the user wants upheld will require a completely different methodology. We (or chat) will somehow need to extract them, and there's no guarantee that it will be done properly.

Each node will display some amount of detail on the node itself, with all of the detail available upon clicking on the node.

### Immediate: 
 - The user is only interested in what the first step would be
 - We would only display the the root node (and maybe the immediate next node)
 - A lot of information regarding the course/certificate

### Medium:
 - The user is interested in the first few steps - for example, the initial course they want and the corresponding certificate
 - We would display the root node, the next node(s) they need to complete to finish that level, and maybe the next course they could take
 - Some detail for each course/certificate

### Broad: 
 - The user is interested in almost or entirely the whole trajectory
 - We would display the root node and every following node until the graph is complete
 - Detail is only available when clicking on a node.
