from typing import List, Tuple, Union, Dict
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers.util import pytorch_cos_sim
import json
import os
from collections import defaultdict
from sklearn.cluster import KMeans
import requests

class Encoder:
    def __init__(self, model_name: str = 'cointegrated/rubert-tiny2', use_gpu: bool = True):
        if not model_name.strip():
            raise ValueError(f'Название модели должно быть заполнено')
        try:
            device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise RuntimeError(f'Не удалось загрузить модель {model_name} c {device}: Ошибка: {e}')
        
    def encode(self, data: Union[List[str], str]) -> torch.Tensor:
        embeddings = None
        try:
            embeddings = self.model.encode(data, convert_to_tensor=True)
        except Exception as e:
            raise RuntimeError(f'Не удалось преобразовать текстовые данные в векторы {e}') from e
        return embeddings
class RAG:
    def __init__(self, encoder: Encoder):
        #if encoder not in Encoder.encode:
        if not isinstance(encoder, Encoder):
            raise ValueError('кодировщик не является экземпляром Encoder')
        self.documents = None
        self.encoder = encoder

    def fit(self, documents: List[str]):
        if not isinstance(documents, list):
            raise ValueError(f'Документы должны быть списком, получен {type(documents)}')
        if not documents:
            raise ValueError('Список документов не должен быть пустым')
        for i in documents:
            if not isinstance(i, str):
                raise ValueError(f'Документы отсутствуют в {documents}')
            if not i.strip():
                raise ValueError(f'Документы не должны быть пустыми строками')
        self.documents = documents
        try:    
            self.doc_embeddings = self.encoder.encode(documents) 
            #return self.embeddings
        except Exception as e:
            raise RuntimeError(f'Ошибка при кодировании документов {e}') from e
    
    def retrieve(self, query: str, retrieval_limit: int = 5, similarity_threshold: float = 0.5) -> Tuple[List[int], List[str]]:
        query_emb = self.encoder.encode(query)
        
        if not self.documents:
            raise ValueError('Документы еще не были установлены.')
        if retrieval_limit < 1 or retrieval_limit > 10:
            raise ValueError('Предел поиск вне заданного диапазона от 1 до 10')
        if retrieval_limit > len(self.documents):
            raise ValueError(f'Предел поиска превышает количество документов: {len(self.documents)}')
        if similarity_threshold > 1 or similarity_threshold < 0:
            raise ValueError('Threshold должен быть в диапазоне от 0 до 1')
        
        candidates = []
        for i, vector in enumerate(self.doc_embeddings):
            score = pytorch_cos_sim(query_emb, vector).item()
            #max_score = score.max().item()
        
            if score > similarity_threshold:
                candidates.append((i, self.documents[i], score))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = candidates[:retrieval_limit]
        
        indices = [item[0] for item in top_candidates]
        texts = [item[1] for item in top_candidates]

        return indices, texts

    def _create_prompt_template(self, query: str, retrieved_docs: List[str]) -> str:
        prompt = "Instructions: Based on the relevant documents, generate a comprehensive response to the user's query.\n"

        prompt += "Relevant Documents:\n"
        for i, doc in enumerate(retrieved_docs):
            prompt += f"Document {i+1}: {doc}\n"

        prompt += f"User Query: {query}\n"

        return prompt

    def _generate(self, query: str, retrieved_docs: List[str]) -> str:
        """
        Generates a response based on the retrieved documents and query.

        Args:
            query (str): The user query.
            retrieved_docs (List[str]): The list of retrieved documents.

        Returns:
            str: The generated response.

        Pseudo-code:
            - Create a prompt using the query and retrieved documents.
            - Pass the prompt to a text generation model.
            - Retrieve and return the generated response.
        """
        # Create the prompt template
        prompt = self._create_prompt_template(query, retrieved_docs)    

        generated_response = ...  # Replace with actual implementation

        return generated_response
        
    def run(self, query: str) -> str:
        """
        Runs the full RAG pipeline: retrieves documents and generates a response.

        Args:
            query (str): The user query.

        Returns:
            str: The generated response.
        """
        _, retrieved_docs = self.retrieve(query)
        generated_response = self._generate(query, retrieved_docs)

        return generated_response

class RAGEval:
    def __init__(
        self,
        documents_path: str,
        questions_path: str,
        retrieval_limit: int = 5,
        similarity_threshold: float = 0.5
    ):      
        self.documents = self.load_documents(documents_path)
        #print("Загруженные документы:", self.documents)
        self.questions = self.load_questions(questions_path)
        self.retrieval_limit = retrieval_limit
        self.similarity_threshold = similarity_threshold

        if not self.documents:
            raise ValueError('Документы отсутствуют')
        if not self.questions:
            raise ValueError("Вопросы отсутствуют")
            
    def load_documents(self, path: str) -> List[str]:
        if not path:
            raise ValueError('Путь к файлу документов не указан')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)    
        except FileNotFoundError:
            raise FileNotFoundError('Файл документов не найден')
        except json.JSONDecodeError as e:
            raise ValueError(f"Файл {path} содержит некорректный JSON: {e}")
            
        if not isinstance(data, list):
            raise ValueError(f"Файл {data} должен содержать список, получен {type(data)}")
        
        documents = []
        for item in data:
            text = self.validate_document(item)
            documents.append(text)
        return documents
    
    def validate_document(self, doc) -> str:
        if not isinstance(doc, dict):
            raise ValueError(f"Документ должен быть словарём, получен {type(doc)}")
        if 'content' not in doc:
            raise ValueError(f"Документ не содержит ключа 'content': {doc}")
            
        content = doc['content']
        
        if not isinstance(content, str):
            raise ValueError(f"Содержимое документа должно быть строкой, получен {type(content)}")
        return content
    
    def load_questions(self, path: str) -> List[str]:
        if not path:
            raise ValueError('Путь к файлу запросов не указан')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)    
        except FileNotFoundError:
            raise FileNotFoundError('Файл c документами не найден')    
        except json.JSONDecodeError as e:
            raise ValueError(f"Файл {path} содержит некорректный JSON: {e}")
            
        if not isinstance(data, list):
            raise ValueError(f"Файл {data} должен содержать список, получен {type(data)}")
        
        questions = [self.validate_question(q) for q in data]
        return questions
    
    def validate_question(self, question) -> str:
        if not isinstance(question, dict):
            raise ValueError(f"Документ должен быть словарём, получен {type(question)}")
        if 'question' not in  question:
            raise ValueError(f"Документ не содержит ключа 'question': {question}")
        
        text_question = question['question']
        return text_question
        
    def evaluate(self, threshold: int = 1) ->Tuple[float, List[int], List[int]]:
        encoder = Encoder()
        rag = RAG(encoder)
        rag.fit(self.documents)
        questions_wo_docs = []
        result_indices_docs = []
        
        for i, query in enumerate(self.questions):
            indices, _  = rag.retrieve(query, self.retrieval_limit, self.similarity_threshold) 
            if len(indices) >= threshold:
                result_indices_docs.extend(indices)
            if not indices:
                questions_wo_docs.append(i) 
        # список индексов документов, которые ни разу не были найдены ни по одному вопросу
        all_docs = set(range(len(self.documents)))
        used_indices = set(result_indices_docs)
        useless_docs = list(all_docs - used_indices)
        number_of_unused = len(useless_docs) / len(self.documents)
        
        # нерелевантные вопросы 
        query_of_unused = len(questions_wo_docs) / len(self.questions)
        rag_score = 1 - number_of_unused - query_of_unused
        return round(rag_score, 2), useless_docs, questions_wo_docs
    
    def clear_docs(self, useless_docs: List[int], output_path: str):
        cleared_docs_idx = []
        for i in self.all_docs:
            if i in useless_docs:
                continue
            else:
                cleared_docs_idx.append(i) 
                
        cleared_docs_text = [{'content': self.documents[i]} for i in cleared_docs_idx]   
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleared_docs_text, f, ensure_ascii=False)

eval = RAGEval('documents_26_02.json', 'questions.json', retrieval_limit=5, similarity_threshold=0.6)
rag_score, useless_docs, questions_wo_docs_idx = eval.evaluate()
eval.clear_docs(useless_docs, 'documents_26_02.json')

class QCluster:
    def __init__(self, questions_idx: List[int], questions: List[str]):
        if len(questions_idx) != len(questions):
            raise ValueError('Списки вопросов и их индексов разной длины')
        if not questions_idx or not questions:
            raise ValueError(f'{questions_idx} или {questions} пуст')
        self.questions_idx = questions_idx
        self.questions = questions
        # заводим словарь для быстрого доступа к вопросам по их индексам
        self.idx_to_question = dict(zip(questions_idx, questions))

    def cluster(self, n_clusters: int, show_results: bool = False) -> Dict[int, List[int]]:
        if n_clusters < 1 or n_clusters > 10:
            raise ValueError(f'Количество {n_clusters} не должно превышать 10')
        if n_clusters > len(self.questions):
            raise ValueError(f'Количество кластеров {n_clusters} превышает число вопросов {len(self.questions)}')
    
        encoder = Encoder()
        embeddings = encoder.encode(self.questions).cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
        
        labels = kmeans.labels_.tolist()
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(self.questions_idx[idx])
        
        sorted_cluster = sorted(clusters.items(), key=lambda item: item[0])
        
        if show_results:
            self.print_clusters(sorted_cluster)
            
        return dict(sorted_cluster)
        
    def print_clusters(self, sorted_cluster):
        for cluster_num, indices in sorted_cluster:
            print(f"Кластер {cluster_num}:")
            for idx in indices:
                question = self.idx_to_question.get(idx)
                print(f'- {question} (Индекс: {idx})')
            print()

questions_wo_docs = [eval.questions[i] for i in questions_wo_docs_idx]
qcluster = QCluster(questions_wo_docs_idx, questions_wo_docs)
clusters = qcluster.cluster(n_clusters=3, show_results=True)

class DocumentGenerator:
    """
    Класс для генерации документов с помощью YandexGPT на основе кластеров вопросов.
    """
    def __init__(self, folder_id: str = None, api_key: str = None):
        """
        Инициализация с параметрами аутентификации YandexGPT.
        Args:
            folder_id: Идентификатор каталога в Yandex Cloud
            api_key: API-ключ сервисного аккаунта
        """
        self.folder_id = folder_id or os.getenv("YC_FOLDER_ID")
        self.api_key = api_key or os.getenv("YC_API_KEY")
        
        if not self.folder_id or not self.api_key:
            raise ValueError("Необходимо указать folder_id и api_key")
        
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.model_uri = f"gpt://{self.folder_id}/yandexgpt-lite/latest"
    
    def generate_document_for_cluster(self, cluster_questions: List[str]) -> str:
        """
        Генерирует документ для кластера вопросов.
        Args:
            cluster_questions: Список вопросов в кластере    
        Returns:
            str: Сгенерированный текст документа
        """
        # Формируем промпт на основе вопросов кластера
        questions_text = "\n".join([f"- {q}" for q in cluster_questions])
        
        prompt = f"""
        Твоя задача — написать единый связный текст, который отвечает на все перечисленные ниже вопросы.  
        **Формат вывода:**  
        - Текст должен быть сплошным (без символов новой строки \\n, без Markdown-разметки типа ###, ** и т.п.).  
        - Предложения разделяются только точками и пробелами.  
        - В конце обязательно поставь точку.  
        - Не копируй вопросы, просто создай полезное содержание.
        
        Вопросы:
        {questions_text}
        
        Напиши текст сейчас в формате энциклопедической статьи или руководства следуя правилам выше.
        """
        # Формируем запрос к YandexGPT
        messages = [
            {
                "role": "system",
                "text": "Ты — полезный ассистент, который создает качественные информационные документы на русском языке."
            },
            {
                "role": "user",
                "text": prompt
            }
        ]
        
        request_body = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": 0.4,  # Немного креативности, но не слишком
                "maxTokens": 250     # Ограничиваем длину ответа
            },
            "messages": messages
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Api-Key {self.api_key}"
                },
                json=request_body,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result['result']['alternatives'][0]['message']['text']
            return generated_text.strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к YandexGPT: {e}")
            return f""
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Ошибка при обработке ответа: {e}")
            return f""
    
    def generate_for_all_clusters(self, clusters: Dict[int, List[int]], 
                                  idx_to_question: Dict[int, str]) -> List[Dict[str, str]]:
        """
        Генерирует документы для всех кластеров.
        
        Args:
            clusters: Словарь {номер_кластера: [индексы_вопросов]}
            idx_to_question: Словарь {индекс_вопроса: текст_вопроса}
            
        Returns:
            List[Dict[str, str]]: Список сгенерированных документов в формате [{"content": "текст"}, ...]
        """
        generated_docs = []
        
        for cluster_num, question_indices in clusters.items():
            print(f"Обработка кластера {cluster_num}...")
            
            # Получаем тексты вопросов для этого кластера
            cluster_questions = [
                idx_to_question[idx] for idx in question_indices 
                if idx in idx_to_question
            ]
            
            if not cluster_questions:
                print(f"  Вопросы для кластера {cluster_num} не найдены, пропускаем")
                continue
            
            print(f"  Найдено вопросов: {len(cluster_questions)}")
            
            # Генерируем документ (можно сгенерировать несколько, если нужно)
            doc_text = self.generate_document_for_cluster(cluster_questions)
            
            if doc_text:
                generated_docs.append({"content": doc_text})
                print(f"  Документ сгенерирован (длина: {len(doc_text)} символов)")
            else:
                print(f"  Не удалось сгенерировать документ для кластера {cluster_num}")
            
            # Небольшая пауза между запросами
            import time
            time.sleep(1)
        
        return generated_docs


def add_documents_to_json(json_path: str, new_docs: List[Dict[str, str]]) -> None:
    """
    Добавляет сгенерированные документы в JSON-файл.
    
    Args:
        json_path: Путь к JSON-файлу с документами
        new_docs: Список новых документов [{"content": "текст"}, ...]
    """
    try:
        # Читаем существующие документы
        with open(json_path, 'r', encoding='utf-8') as f:
            existing_docs = json.load(f)
        
        # Добавляем новые документы
        existing_docs.extend(new_docs)
        
        # Записываем обратно
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_docs, f, ensure_ascii=False, indent=4)
        
        print(f"Добавлено {len(new_docs)} новых документов. Всего документов: {len(existing_docs)}")
        
    except FileNotFoundError:
        print(f"Файл {json_path} не найден")
    except json.JSONDecodeError as e:
        print(f"Ошибка при чтении JSON: {e}")
    
generator = DocumentGenerator(
    folder_id='*************************************',  # Замените на ваш folder_id
    api_key='*************************************'  # Замените на ваш api_key
)

new_documents = generator.generate_for_all_clusters(clusters, qcluster.idx_to_question)

add_documents_to_json('documents_26_02.json', new_documents)