 We will be designing the experiments for templatic in-context reasoning tasks that require associating together groups of entities and answering questions about them.

 Here is the conceptual framework for an entity binding task:
  
\begin{enumerate}
    \item \textbf{Entity Roles}: Disjoint sets of entities $\mathcal{E}_1,\dots,\mathcal{E}_m$ that will fill particular roles in a templatic text. For example, the set $\mathcal{E}_1$ might be names of people $\{ \textit{Ann}, \textit{Pete}, \textit{Tim}, \dots \}$, and the set $\mathcal{E}_2$ might be foods and drinks $\{ \textit{ale}, \textit{jam}, \textit{pie}, \dots \}$. 
    \item \textbf{Entity Groups}: An entity group is a tuple $G \in \mathcal{E}_1 \times \dots \times \mathcal{E}_m$ containing entities that will be placed within the same clause in a template. For example, we could set $G_1=(\textit{Pete}, \textit{jam})$ and $ G_2=(\textit{Ann}, \textit{pie})$. For convenience, we define $\mathbf{G}$ as a binding matrix wherein $\mathbf{G}_i^j$ denotes the $j$-th entity in the $i$-th entity group.
    \item A \textbf{template} ($\mathcal{T}$): A function that takes as input a binding matrix $\mathbf{G}$, the \textit{query entity} $q = \mathbf{G}^{\qg}_{\qe}$, and the target entity $t = \mathbf{G}^{\qg}_{\te}$. Here $\qg$ is a positional index of the entity group containing the target and query, and $\te \not = \qe$ index the positions of the target and query entities within that group, respectively. See \S\ref{appx:binding_tasks} for more details and examples.
\end{enumerate}
    Continuing our example, define 
    \[\mathcal{T}(\mathbf{G}, q, t) = G^{1}_1 \textit{ loves } G^2_1, G^{1}_2 \textit{ loves } G^2_2. \begin{cases}\textit{Who loves } q \textit{?} & \te == 1 \\\textit{What does } q \textit{ love?} & \te == 2 \\ \end{cases} \\ \]
    and observe that 
    \[\mathcal{T}\Big(\begin{bmatrix}
\textit{Pete} & \textit{jam} \\
\textit{Ann}    & \textit{pie}
\end{bmatrix}, \textit{pie}, \textit{Ann}\Big)= \textit{Pete loves jam, Ann loves pie. Who loves pie?}\]

For our experiments, the binding matrix $\mathbf{G}$ will consist of distinct entities.

A natural bundle of information is a template for stating the entity relations, a set of question templates for each subset of indices, a set of fillers for each group index. This is enough to generate a specific dataset. Also include a map from group indices to intelligible names like "human subject" and "thing loved".

Think about what primitives you will need to write code that will support the flexible generation of tasks with an arbitrary number of binding groups and an arbitrary number of entities in each group. 

## Behavioral Causal Model
The simplest causal model will need variables for the input entities of each group. To have one causal model that can handle an input with up to *k* entity groups and up to d entities within a group, we will need create that many variables, and let variables be dead or inactive for inputs that have less than *k* groups of less than *d* entities.

also template variables and question templates variable that are where templates are fed in. The question templates will store all of the templates in a dictionary mapping from orderded subsets of query indices paired with the answer index to templates. The output will be the answer to the question.

To understand the nature of the question template, consider all the different ways of asking about "Harry put water into the cup" that ask a query entity in a given position using a subset of entities in the other positions

What did Harry put in the cup?\

What was put in the cup by Harry?

Who put water in the cup?\

Who touched the cup?\

What did Harry pour?\

What container was filled by Harry?\

## Input samplers

The input sampler should make sure that the binding matrix contains completely distinct entities and that the question being asked can be answered.

## Positional Causal Model

This causal model will have a single intermediate variable stores the **position** of the group containing the answer. This position is then dereferenced to retrieve the entity from the appropriate position in that group.

When the positional variable is intervened on, this changes which group the answer is retrieved from, but it does not change the position within that group.

## Positional Entity Causal Model

This is an extended version of the positional causal model that makes the position computation more explicit through additional intermediate variables. Rather than directly computing which group contains the query entity, this model breaks the computation into multiple stages:

### Variables

1. **Positional Entity Variables** (`positional_entity_g{g}_e{e}`): For every input entity at group `g`, position `e`, this variable computes the position (group index) that this entity occupies in the list of entity groups. Simply returns `g`.

2. **Positional Query Variables** (`positional_query_e{e}`): For each entity position `e` within a group, this variable:
   - Identifies which entity appears at position `e` in the query (from `query_indices`)
   - Searches through all groups to find which group positions contain that query entity at position `e`
   - Returns a tuple of all matching group positions

   For example, if the query asks about the entity at position 0 in some group, and that entity is "Ann", this variable finds all groups where "Ann" appears at position 0.

3. **Positional Answer Variable** (`positional_answer`):
   - Takes the intersection of all positional query variables for entities mentioned in the query
   - Must result in exactly ONE group position (otherwise throws an error indicating ambiguity)
   - Returns that single group position where the answer should be retrieved from

4. **Raw Output Variable** (`raw_output`):
   - Uses the `positional_answer` variable to determine which group to retrieve from
   - Uses the `answer_index` to determine which position within that group
   - Retrieves and returns the entity value at that location

### Mechanism Example

Consider the prompt: "Pete loves jam, Ann loves pie. What does Ann love?"

- Query: position 0 entity ("Ann"), answer: position 1 entity
- `positional_entity_g0_e0` = 0 (Pete is in group 0)
- `positional_entity_g1_e0` = 1 (Ann is in group 1)
- `positional_query_e0` = (1,) — "Ann" appears at position 0 in group 1
- `positional_answer` = 1 — intersection yields single group 1
- `raw_output` = entity at g1_e1 = "pie"

### Intervention Effects

- **Intervening on `positional_entity_g{g}_e{e}`**: Changes which position a specific entity is considered to be at
- **Intervening on `positional_query_e{e}`**: Changes which groups are considered as candidates for a specific query position
- **Intervening on `positional_answer`**: Directly changes which group the answer is retrieved from



# ⚠️ STOP HERE FOR STEP 1 ⚠️

## Token Positions

Write a function that can be used to index into tokens using:

1. The position of a group of entities in a list of groups
2. The position of an entity within that group
3. The position of tokens for that entity. If the entity is tokenized into multiple tokens, we might select all of the tokens or just the last token.

When an entity appears multiple times in a prompt (like "Pete" appearing in both "Pete loves jam" and "What does Pete love?"), we need to identify which occurrence we're targeting. A naive substring search always finds the first occurrence, creating ambiguity for interventions.

The solution uses template knowledge to parse the prompt into structured regions. The system understands that prompts have a statement region (where facts are stated) and a question region (where the query appears). You can then ask for token positions of an entity specifically in the statement or specifically in the question.

This matters for interventions because intervening on the entity representation in the statement (where it's being encoded) is fundamentally different from intervening on that same entity in the question (where it's being retrieved). The structured parsing ensures you're targeting the right occurrence.

## Counterfactual Datasets

To test this, we generate counterfactuals by swapping entity groups while keeping the entities in the query text fixed. For example, if the input asks about entity group 1 and group 1 contains (Ann, pie), the counterfactual might swap group 1 with group 2, so group 1 now contains (Bob, cake). The query still asks about "entity group 1, position 0" - but a different entity now occupies that position.


## Language Model
```yaml
models:
  - meta-llama/Llama-3.1-8B-Instruct
```

## Token Positions

Use the last token position.

## Experiments

Run an experiment with inputs containing two groups each with two entities of the form:

```
We will ask a question about the following sentences.

Tim loves soup and Pete loves bread.

Who loves bread?
Answer:
```

Sample inputs with only two entity groups and generate a counterfactual dataset by swapping entity groups while keeping the same question. Filter the dataset to include only examples where the model answers correctly.

Run full vector residual stream patching at the last token across all layers including layer -1.

Construct heatmaps for the positional variable and the raw output variable.

Then run that same experiment again with inputs containing three groups each with three entities of the form:

```
We will ask a question about the following sentences.

Kate put key in the shelf, Tim put coin in the pocket, and Sue put book in the bag.

Where was key put?
Answer:
```

## Output File

For each of the two datasets:

- Which two layers have the highest positional signal?
- Which two layers have the lowest positional signal?

{
    "love_positional_high": [$LAYER_0, $LAYER_1]
    "love_output_low": [$LAYER_2, $LAYER_3]
    "put_positional_high": [$LAYER_0, $LAYER_1]
    "put_output_low": [$LAYER_2, $LAYER_3]
}
