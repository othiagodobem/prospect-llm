from types import SimpleNamespace

from llm.provider import LLMProvider
from schemas.artifacts import GeracaoPerguntas


class FakeCompletions:
    def create(self, **kwargs):
        content = (
            '{"roteiro_diagnostico":{"perguntas":['
            '{"tipo":"contexto","pergunta":"Como funciona hoje?",'
            '"prioridade":1,"finalidade":"Entender o processo"},'
            '{"tipo":"impacto","pergunta":"Qual é o impacto?",'
            '"prioridade":2,"finalidade":"Medir consequências"},'
            '{"tipo":"decisao","pergunta":"Quem participa da decisão?",'
            '"prioridade":3,"finalidade":"Mapear decisão"}]},'
            '"alertas_validacao":[]}'
        )
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        return SimpleNamespace(choices=[choice], usage=usage)


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_provider_validates_json():
    provider = LLMProvider(client=FakeClient(), model="deepseek-v4-flash")
    result, metadata = provider.generate_structured(
        "Responda em JSON", "Crie perguntas", GeracaoPerguntas
    )
    assert len(result.roteiro_diagnostico.perguntas) == 3
    assert metadata["modelo"].startswith("deepseek/")
