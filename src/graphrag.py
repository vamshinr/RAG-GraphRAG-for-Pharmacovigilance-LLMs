import pandas as pd
from neo4j import GraphDatabase

class GraphRAGSystem:
    def __init__(self, uri, user, password, data_path='data/drug_side_effects.csv'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.df = pd.read_csv(data_path)
        self._create_graph()
        print("GraphRAG System Initialized and graph populated.")

    def close(self):
        self.driver.close()

    def _create_graph(self):
        """
        Populates the Neo4j database with drugs and side effects.
        This function is idempotent - it won't create duplicate nodes/relationships.
        """
        with self.driver.session() as session:
            # Clear existing graph to avoid duplicates on re-run
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create constraints for uniqueness
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE")
            
            # Populate graph from the dataframe
            for _, row in self.df.iterrows():
                session.run("""
                MERGE (d:Drug {name: $drug_name})
                MERGE (s:SideEffect {name: $side_effect})
                MERGE (d)-[:HAS_SIDE_EFFECT {description: $description}]->(s)
                """, 
                drug_name=row['drug_name'], 
                side_effect=row['side_effect'], 
                description=row['description']
                )

    def search(self, drug_name: str):
        """
        Queries the graph for side effects of a specific drug.
        """
        with self.driver.session() as session:
            result = session.run("""
            MATCH (d:Drug {name: $drug_name})-[r:HAS_SIDE_EFFECT]->(s:SideEffect)
            RETURN s.name AS side_effect, r.description AS description
            """, drug_name=drug_name)
            
            return [f"{record['side_effect']}: {record['description']}" for record in result]

# Example of how to use it
if __name__ == '__main__':
    # --- IMPORTANT: Replace with your Neo4j credentials ---
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "your_password_here" 
    
    graph_rag = GraphRAGSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    drug_query = "Paracetamol"
    retrieved_info = graph_rag.search(drug_query)
    
    print(f"Querying for side effects of: {drug_query}")
    print("\nRetrieved Information:")
    for info in retrieved_info:
        print(f"- {info}")
        
    graph_rag.close()
