# WikiSQL

`Semantic Parsing`, `NL2SQL`

- `WikiSQL`: A large crowd-sourced dataset for developing natural language interfaces for relational databases.

---

## Results

- Column details
	* Agg: Aggregator 
	* Sel: SELECT Column
	* Cond: Where clause
	* LF: Logical Form
	* EX: Execution
	* (): Paper result

| Model | Agg | Sel | Cond | LF | EX | BaseConfig |
| --- | --- | --- | --- | --- | --- | --- |
| **SQLNet** | (90.1) | (91.1) | (72.1) | - | (69.8) | wikisql/sqlnet.json |