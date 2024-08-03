import json
import re
from evaluate import load
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.corpus import wordnet

def load_qa_pairs(file_path):
    """Load QA pairs from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['qa_pairs']

def extract_sentences(text):
    """Extract sentences from text."""
    return [sent.strip() for sent in sent_tokenize(text) if sent.strip()]

def compute_bleu_score(context, generated_answers):
    bleu = load("bleu")
    
    # Each prediction must be compared to multiple references, here we use the same context as references.
    references = [extract_sentences(context)] * len(generated_answers)  # Create a list of references for each prediction
    
    if not references or not generated_answers:
        print("Warning: Empty input for BLEU score computation")
        return {"bleu": 0.0}
    
    try:
        return bleu.compute(predictions=generated_answers, references=references)
    except Exception as e:
        print(f"Error computing BLEU score: {str(e)}")
        return {"bleu": 0.0}

def compute_rouge_score(context, generated_answers):
    rouge = load("rouge")
    
    references = [extract_sentences(context)] * len(generated_answers)  # Create a list of references for each prediction
    
    if not references or not generated_answers:
        print("Warning: Empty input for ROUGE score computation")
        return rouge.compute(predictions=generated_answers, references=references)
    
    return rouge.compute(predictions=generated_answers, references=references)

def compute_exact_match(context, generated_answers):
    sentences = extract_sentences(context)
    
    exact_matches = sum([1 for gen in generated_answers if any(gen in sent for sent in sentences)])
    return exact_matches / len(generated_answers) * 100

def compute_meteor_score(context, generated_answers):
    meteor = load("meteor")
    
    references = [extract_sentences(context)] * len(generated_answers)  # Create a list of references for each prediction
    
    return meteor.compute(predictions=generated_answers, references=references)

def compute_question_difficulty(questions):
    """Estimate question difficulty based on length and complexity."""
    difficulties = []
    for q in questions:
        length = len(q.split())
        # Flatten the list of synsets and count unique ones
        synsets = [synset for word in q.split() for synset in wordnet.synsets(word)]
        complexity = len(set(synsets))  # Unique synsets
        difficulty = length * complexity
        difficulties.append(difficulty)
    
    return {
        "average_difficulty": sum(difficulties) / len(difficulties) if difficulties else 0,
        "difficulty_distribution": Counter(difficulties)
    }


def compute_answer_informativeness(answers, context):
    """Estimate answer informativeness based on coverage of context."""
    sentences = set(extract_sentences(context))
    
    informativeness = []
    for ans in answers:
        words_in_ans = set(ans.split())
        coverage = len(words_in_ans.intersection(sentences)) / len(sentences) * 100
        informativeness.append(coverage)
    
    return {
        "average_informativeness": sum(informativeness) / len(informativeness) if informativeness else 0
    }

def compute_question_relevance(questions, context):
    """Estimate question relevance based on overlap with context."""
    sentences = set(extract_sentences(context))
    
    relevances = []
    for q in questions:
        words_in_q = set(q.split())
        overlap = len(words_in_q.intersection(sentences)) / len(sentences) * 100
        relevances.append(overlap)
    
    return {
        "average_relevance": sum(relevances) / len(relevances) if relevances else 0
    }

def evaluate_qa_pairs(context, generated_qa_pairs):
    generated_questions = [qa['question'] for qa in generated_qa_pairs]
    generated_answers = [qa['answer'] for qa in generated_qa_pairs]

    # Evaluate answers
    bleu_score = compute_bleu_score(context, generated_answers)
    rouge_score = compute_rouge_score(context, generated_answers)
    exact_match_score = compute_exact_match(context, generated_answers)
    meteor_score = compute_meteor_score(context, generated_answers)
    
    # Evaluate questions
    bleu_question_score = compute_bleu_score(context, generated_questions)
    rouge_question_score = compute_rouge_score(context, generated_questions)
    exact_match_question_score = compute_exact_match(context, generated_questions)
    meteor_question_score = compute_meteor_score(context, generated_questions)
    
    # Additional Metrics
    question_difficulty = compute_question_difficulty(generated_questions)
    answer_informativeness = compute_answer_informativeness(generated_answers, context)
    question_relevance = compute_question_relevance(generated_questions, context)
    
    # Results
    results = {
        "bleu_score_answers": bleu_score,
        "rouge_score_answers": rouge_score,
        "exact_match_score_answers": exact_match_score,
        "meteor_score_answers": meteor_score,
        "bleu_score_questions": bleu_question_score,
        "rouge_score_questions": rouge_question_score,
        "exact_match_score_questions": exact_match_question_score,
        "meteor_score_questions": meteor_question_score,
        "question_difficulty": question_difficulty,
        "answer_informativeness": answer_informativeness,
        "question_relevance": question_relevance,
    }
    
    return results

def main():
    context = """Enter your context here. Ensure that this context is sufficiently detailed to generate meaningful QA pairs."""
    generated_qa_pairs = load_qa_pairs("generated_qa_pairs.json")

    evaluation_results = evaluate_qa_pairs(context, generated_qa_pairs)
    
    print("Evaluation Results:")
    print(f"BLEU Score (Answers): {evaluation_results['bleu_score_answers']}")
    print(f"ROUGE Score (Answers): {evaluation_results['rouge_score_answers']}")
    print(f"Exact Match Score (Answers): {evaluation_results['exact_match_score_answers']}%")
    print(f"METEOR Score (Answers): {evaluation_results['meteor_score_answers']}")
    print(f"BLEU Score (Questions): {evaluation_results['bleu_score_questions']}")
    print(f"ROUGE Score (Questions): {evaluation_results['rouge_score_questions']}")
    print(f"Exact Match Score (Questions): {evaluation_results['exact_match_score_questions']}%")
    print(f"METEOR Score (Questions): {evaluation_results['meteor_score_questions']}")
    print(f"Question Difficulty: {evaluation_results['question_difficulty']}")
    print(f"Answer Informativeness: {evaluation_results['answer_informativeness']}")
    print(f"Question Relevance: {evaluation_results['question_relevance']}")

if __name__ == "__main__":
    main()

