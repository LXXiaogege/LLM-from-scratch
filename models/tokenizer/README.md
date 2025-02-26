# Tokenizer 文件说明

在大语言模型的训练和推理中，`Tokenizer` 负责将文本转换为模型可以理解的输入格式。以下是常见的 Tokenizer 文件及其功能：

## 1. `merges.txt`
- **用途**：存储 **Byte Pair Encoding (BPE)** 合并规则。BPE 是一种子词分词方法，通过迭代地合并最频繁的字符对来生成子词单元。
- **内容**：每行表示一个字符对，指示两个最常见的字符或子词需要合并成一个新子词。例如，`l o` 表示字符 `l` 和 `o` 可以合并成一个新的子词 `lo`。
- **用途示例**：如果文本中有 `low`，且 `merges.txt` 中有合并规则 `l o -> lo`，则 `low` 会被拆分为 `lo` 和 `w`。

## 2. `tokenizer.json`
- **用途**：包含 Tokenizer 的完整配置文件，包含分词算法、词表等信息。它保存了如何将文本转化为 token 的所有规则。
- **内容**：包括与 `merges.txt` 和 `vocab.json` 的关联配置，分词方法等信息。它帮助 Tokenizer 初始化并正确处理输入文本。
- **用途示例**：加载预训练模型时，`tokenizer.json` 提供分词和解码所需的所有配置和信息。

## 3. `tokenizer_config.json`
- **用途**：存储与 Tokenizer 本身的配置相关的元数据，例如分词方式、是否使用特殊标记（如 [CLS]、[SEP]）以及模型的其他设置。
- **内容**：包含 Tokenizer 行为的配置，如 `do_lower_case`、`add_special_tokens` 等参数，决定如何处理文本。
- **用途示例**：例如，`"do_lower_case": true` 表示分词时将文本转换为小写。

## 4. `vocab.json`
- **用途**：包含词汇表（Vocabulary），是 token 与索引的映射字典。每个 token（字符、子词、或单词）都对应一个唯一的整数索引。
- **内容**：文件是一个 JSON 字典，键是 token（如 `hello`、`low`、`##ing`），值是该 token 对应的索引（如 `12345`）。
- **用途示例**：例如，`"hello": 12345` 表示文本 "hello" 会被映射为索引 `12345`，这是模型所理解的格式。

---

## 总结
- **`merges.txt`**：存储 BPE 合并规则，用于子词分词。
- **`tokenizer.json`**：包含 Tokenizer 的完整配置信息。
- **`tokenizer_config.json`**：包含 Tokenizer 的元配置，如特殊标记和行为设置。
- **`vocab.json`**：包含词汇表，是 token 和索引之间的映射字典。
