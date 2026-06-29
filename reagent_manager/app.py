import streamlit as st
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

# --- 1. 核心常量定义 (确保函数内能访问) ---
STANDARD_COLS = ["库存地", "位置/格子", "中文名", "英文名", "CAS号", "规格", "品牌", "数量"]
MASTER_FILE = "试剂总库_不要删除.xlsx"

# 列名模糊匹配字典
MAPPING = {
    "库存地": ["库存地", "地点", "存放点", "柜子"],
    "位置/格子": ["位置", "格子", "位置/格子", "序号", "冰", "柜号"],
    "中文名": ["中文名", "名称", "药品名", "试剂名", "中文名称"],
    "英文名": ["英文名", "英文名称", "Name", "English Name"],
    "CAS号": ["CAS号", "CAS", "CAS NO"],
    "规格": ["规格", "含量", "包装", "容量"],
    "品牌": ["品牌", "厂家", "生产商"],
    "数量": ["数量", "剩余量", "个数", "单位"]
}

# 地点自动归一化规则
MAPPING_RULES = {
    "C3a 710冰箱冷藏柜": ["冰箱", "冰", "C2b 428"],
    "C3a 710电子防潮柜": ["电子防潮柜", "电子柜"],
    "C3a 710防潮柜": ["防潮柜"],
    "C3a 710药品柜": ["药品柜"],
    "酸碱柜": ["酸柜", "碱柜", "酸碱"],
    "七楼总库": ["七楼", "总清单", "实验室总库"]
}

st.set_page_config(page_title="实验室试剂管理系统", layout="wide")


# --- 2. 辅助函数 ---
def select_file(is_save=False):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    if is_save:
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    else:
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    root.destroy()
    return path


def normalize_location(name):
    name = str(name).strip()
    for std_name, keywords in MAPPING_RULES.items():
        if any(k in name for k in keywords):
            return std_name
    return name


def smart_load(file_path):
    try:
        file_name = os.path.basename(file_path).split('.')[0]
        # 文件名仅作为“最后走投无路”时的保底地点
        file_default_loc = file_name.replace("药物清单", "").replace("试剂清单", "").strip()

        all_sheets = pd.read_excel(file_path, sheet_name=None, header=None, dtype=str)
        cleaned_list = []

        for sheet_name, df in all_sheets.items():
            if df.empty: continue

            # 1. 精准定位表头行
            header_idx = 0
            for i, row in df.iterrows():
                row_str = "".join(row.fillna("").astype(str))
                if "中文名" in row_str or "CAS" in row_str:
                    header_idx = i
                    break

            df.columns = [str(c).strip() for c in df.iloc[header_idx]]
            df = df.iloc[header_idx + 1:].reset_index(drop=True)

            temp_df = pd.DataFrame(columns=STANDARD_COLS)

            # 2. 识别列（特别注意：看原表有没有“库存地”这一列）
            has_original_loc = any(c in MAPPING["库存地"] for c in df.columns)

            for std_col, aliases in MAPPING.items():
                for col in df.columns:
                    if col in aliases:
                        temp_df[std_col] = df[col]
                        break

            # 3. 【核心修正】差异化地点处理
            def assign_location(row):
                # 提取原表中填写的地点
                raw_val = str(row["库存地"]).strip() if has_original_loc else ""

                # 如果原表里有具体的地点（不是空的且不是nan），就用原表的
                if raw_val and raw_val.lower() != "nan" and raw_val != "":
                    return normalize_location(raw_val)

                # 只有原表没这一列或没写地点时，才用文件名补全（比如冰箱分清单）
                return normalize_location(file_default_loc)

            temp_df["库存地"] = temp_df.apply(assign_location, axis=1)

            # 4. 解决总清单特有的“地点+格子”连体问题 (如: C3a 710电子防潮柜 A1)
            def split_loc_and_pos(row):
                loc = str(row["库存地"])
                pos = str(row["位置/格子"])
                if (pos == "" or pos == "nan") and " " in loc:
                    parts = loc.rsplit(" ", 1)
                    return parts[0], parts[1]
                return loc, pos

            temp_df[["库存地", "位置/格子"]] = temp_df.apply(lambda r: pd.Series(split_loc_and_pos(r)), axis=1)

            # 过滤掉杂质行
            temp_df = temp_df[temp_df["中文名"].notna() & (temp_df["中文名"] != "")]
            cleaned_list.append(temp_df)

        return pd.concat(cleaned_list, ignore_index=True)
    except Exception as e:
        st.error(f"解析失败: {e}")
        return None


# --- 3. UI 界面逻辑 ---
st.title("🧪 Lin Group实验室试剂智能管理系统")

with st.sidebar:
    st.header("文件操作")
    if st.button("📁 导入/合并旧 Excel"):
        path = select_file()
        if path:
            new_data = smart_load(path)
            if new_data is not None:
                if os.path.exists(MASTER_FILE):
                    old_data = pd.read_excel(MASTER_FILE, dtype=str)
                    combined = pd.concat([old_data, new_data], ignore_index=True).drop_duplicates()
                else:
                    combined = new_data
                combined.to_excel(MASTER_FILE, index=False)
                st.success("数据已导入并自动对齐地点！")
                st.rerun()

    if st.button("💾 导出当前分类 (用于打印)"):
        if 'current_df' in st.session_state and not st.session_state.current_df.empty:
            save_path = select_file(is_save=True)
            if save_path:
                st.session_state.current_df.to_excel(save_path, index=False)
                st.success(f"已导出至: {save_path}")

# --- 4. 主显示与编辑区 ---
if os.path.exists(MASTER_FILE):
    df = pd.read_excel(MASTER_FILE, dtype=str).fillna("")

    # 过滤器：按柜子筛选
    locs = ["全部显示"] + sorted(df["库存地"].unique().tolist())
    sel_loc = st.selectbox("选择要管理的柜子/冰箱：", locs)

    display_df = df if sel_loc == "全部显示" else df[df["库存地"] == sel_loc]
    st.session_state.current_df = display_df

    st.subheader(f"📍 位置: {sel_loc}")
    # 在线编辑功能：支持直接删除行、增加行、修改格子内容
    edited_df = st.data_editor(display_df, num_rows="dynamic", use_container_width=True)

    if st.button("🚀 提交修改（保存增加/删除/去掉的结果）"):
        # 这一步是把当前网页上“剩下的”数据写回文件
        if sel_loc == "全部显示":
            final_df = edited_df
        else:
            # 如果是筛选状态，要保留其他柜子的药，合并当前修改的药
            others = df[df["库存地"] != sel_loc]
            final_df = pd.concat([others, edited_df], ignore_index=True)

        # 强制覆盖保存
        final_df.to_excel(MASTER_FILE, index=False)
        st.success("清理完毕！文件已重写。")
        st.rerun()  # 强制刷新页面