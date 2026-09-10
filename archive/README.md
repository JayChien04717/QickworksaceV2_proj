# 原始版本封存

`v1/` 保留本次重建前的 package、notebooks、tutorial、example 和根目錄說明／安裝檔案。它們不會被新版 package 匯入或發行；保留供比較、回復和移植專用量測。

不要將 archive 加到 sys.path 來混用新舊 module。舊 notebook 若需要重現，應使用獨立的舊環境，或依 docs/MIGRATION.md 移植。
