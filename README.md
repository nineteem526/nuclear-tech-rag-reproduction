# 企业技术文档 RAG 复现项目

这是基于既有项目经验重新设计和实现的学习型复现项目，当前配置不代表原项目的线上配置。

## 当前增量：Phase 1 / Task 1

当前只实现：

```text
文字型 PDF -> ParsedDocument -> DocumentBlock -> Chunk
```

明确不包含 Embedding、Elasticsearch、BM25、Rerank、LLM、OCR 或 Kafka。

## 安装

```powershell
uv sync --extra dev --no-editable
```

项目路径包含中文。这里使用非 editable 安装，避免部分 Windows/Python 组合读取 editable `.pth` 路径时发生编码错位。

## 运行

```powershell
uv run --no-sync rag-pdf parse-chunk .\path\document.pdf
```

自定义分块参数并保存 JSON：

```powershell
uv run --no-sync rag-pdf parse-chunk .\path\document.pdf `
  --target-tokens 512 `
  --overlap-tokens 80 `
  --output .\runtime\parse-result.json
```

## 测试

```powershell
uv run --no-sync pytest
```

## 当前 token 计数边界

这一增量使用显式实现的混合文本近似 token 计数器：技术编号作为整体 token，中文字符逐字计数。它保证测试可重复，但不声称与 BGE-M3 tokenizer 完全一致。Embedding 阶段接入模型 tokenizer 后，可以通过 `TokenCounter` 接口替换，而不用修改 Chunker 主算法。

