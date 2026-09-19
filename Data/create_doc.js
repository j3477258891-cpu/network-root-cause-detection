const docx = require('docx');
const fs = require('fs');
const { Document, Paragraph, TextRun, HeadingLevel, AlignmentType, PageBreak } = docx;

const doc = new Document({
    sections: [{
        properties: {
            page: {
                size: { width: 12240, height: 15840 }
            }
        },
        children: [
            new Paragraph({
                text: "（注：代码复核评审需要登录选手的账号进行复核及测试）",
                alignment: AlignmentType.CENTER,
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "网络故障根因分析团队代码使用文档",
                heading: HeadingLevel.HEADING_1,
                alignment: AlignmentType.CENTER,
                spacing: { after: 400 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "比赛平台账号：", bold: true }),
                    new TextRun("rootcause_analysis_team")
                ],
                spacing: { after: 200 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "密码：", bold: true }),
                    new TextRun("********")
                ],
                spacing: { after: 200 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "团队成员：", bold: true }),
                    new TextRun("张三、李四、王五")
                ],
                spacing: { after: 200 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "联系电话：", bold: true }),
                    new TextRun("138****8888")
                ],
                spacing: { after: 400 }
            }),

            new Paragraph({
                text: "1. 算法思路（主要基于什么算法，是如何实现的）",
                heading: HeadingLevel.HEADING_2,
                spacing: { before: 400, after: 200 }
            }),

            new Paragraph({
                text: "1.1 数据层面：",
                heading: HeadingLevel.HEADING_3,
                spacing: { before: 200, after: 200 }
            }),

            new Paragraph({
                text: "数据预处理的步骤如下：",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "1、数据加载与解析：从训练集和测试集中加载网络拓扑数据（.log.topo.json）和根因标签数据（.rootcause.json）。每个案例包含完整的网络拓扑图结构，节点代表网络设备（如基站、传输电路、设备等），边代表设备间的依赖关系。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "2、数据清洗：对加载的拓扑数据进行清洗，处理缺失值和异常节点。验证节点的依赖关系完整性，确保入边和出边的一致性。清理孤立节点和无效的依赖关系，提高数据质量。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "3、图构建：将清洗后的数据构建成有向图结构。使用NetworkX库构建图，节点保存设备的属性信息（类别、供应商、设备类型等），边表示设备间的依赖关系。构建时保持图的完整性和准确性。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "4、特征提取：针对每个节点提取多维度特征，包括：结构特征（度数、邻居数、聚类系数）、中心性特征（度中心性、介数中心性、PageRank）、属性特征（设备类别、供应商、设备类型的编码）。同时提取图级别特征（节点数、边数、密度、连通性等）。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "5、特征标准化：对提取的特征进行标准化处理，使用Z-score标准化使特征具有相同的尺度和范围。标准化可以提高模型的训练效果和收敛速度。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "6、数据格式转换：将处理后的数据转换成适合模型输入的格式，保存为pickle格式以便后续快速加载。",
                spacing: { after: 300 }
            }),

            new Paragraph({
                text: "1.2 算法层面：",
                heading: HeadingLevel.HEADING_3,
                spacing: { before: 200, after: 200 }
            }),

            new Paragraph({
                text: "本方案采用集成学习方法进行根因预测，主要思路如下：",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "1、问题建模：将根因分析问题建模为二分类问题，对图中的每个节点预测其是否为故障根因。通过计算每个节点是故障源的概率，选择概率最高的Top-K个节点作为预测的根因。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "2、特征工程：基于图结构提取节点的结构特征、中心性特征和属性特征。结构特征反映节点在网络中的连接模式，中心性特征衡量节点的重要程度，属性特征包含设备的类型和供应商信息。这些多维度特征能够全面刻画节点的特性。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "3、集成学习模型：采用随机森林（Random Forest）和梯度提升（Gradient Boosting）两种集成学习算法。随机森林通过构建多个决策树并投票来提高预测的稳定性和泛化能力；梯度提升通过逐步优化损失函数来提高预测精度。两个模型的预测概率进行加权平均，得到最终的集成预测结果。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "4、类别不平衡处理：由于故障节点在整个网络中占比很小，存在严重的类别不平衡问题。采用类别权重调整（class_weight='balanced'）和样本权重（sample_weight）两种方法来解决。通过提高正样本的权重，使模型更加关注少数类样本。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "5、预测策略：对测试集中的每个案例，计算所有节点的故障概率。选择概率最高的Top-K个节点（动态调整K值，通常为节点总数的1%或至少1个至多5个）作为根因预测结果。设置概率阈值（0.3）过滤低置信度的预测，提高预测准确性。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "6、结果后处理：根据预测的节点信息，生成符合提交格式的根因报告。包括故障节点的ID、故障标题、位置信息和可能的故障原因。结合设备类型和历史故障模式，自动生成合理的故障描述。",
                spacing: { after: 300 }
            }),

            new Paragraph({
                text: "2. 模型架构说明",
                heading: HeadingLevel.HEADING_2,
                spacing: { before: 400, after: 200 }
            }),

            new Paragraph({
                text: "整体模型架构包含6个模块，从数据获取到最终预测结果输出形成完整的流水线：",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "模块1：数据获取和清洗模块",
                spacing: { after: 100 }
            }),
            new Paragraph({
                text: "- 功能：加载原始拓扑数据和标签数据，进行数据清洗和验证",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输入：train和test文件夹中的.log.topo.json和.rootcause.json文件",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输出：清洗后的图结构数据，保存为pickle格式",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "模块2：图构建模块",
                spacing: { after: 100 }
            }),
            new Paragraph({
                text: "- 功能：将清洗后的数据构建成NetworkX有向图结构",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输入：清洗后的节点和边数据",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输出：NetworkX图对象，包含完整的节点属性和边关系",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "模块3：特征提取模块",
                spacing: { after: 100 }
            }),
            new Paragraph({
                text: "- 功能：提取节点的多维度特征和图级别特征",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输入：构建好的图结构",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输出：特征向量矩阵，包括结构特征、中心性特征和属性特征",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "模块4：模型训练模块",
                spacing: { after: 100 }
            }),
            new Paragraph({
                text: "- 功能：训练随机森林和梯度提升两个集成学习模型",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输入：训练集的特征向量和标签",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输出：训练好的模型文件和特征标准化器",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "模块5：预测模块",
                spacing: { after: 100 }
            }),
            new Paragraph({
                text: "- 功能：使用训练好的模型对测试集进行根因预测",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输入：测试集的特征向量和训练好的模型",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输出：每个测试案例的根因节点预测结果和置信度",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "模块6：结果格式化模块",
                spacing: { after: 100 }
            }),
            new Paragraph({
                text: "- 功能：将预测结果转换为符合提交要求的格式",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输入：预测结果和测试集原始数据",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 输出：标准格式的.rootcause.json文件，包含故障节点的详细信息",
                spacing: { after: 300 }
            }),

            new Paragraph({
                text: "3. 代码使用方法（重要）",
                heading: HeadingLevel.HEADING_2,
                spacing: { before: 400, after: 200 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "代码存放位置：", bold: true }),
                    new TextRun("./（当前目录）")
                ],
                spacing: { after: 200 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "代码文件列表：", bold: true })
                ],
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "- 01_data_preprocess.py：数据预处理脚本",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 02_feature_extraction.py：特征提取脚本",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 03_model_training.py：模型训练脚本",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 04_model_prediction.py：模型预测脚本",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 05_format_results.py：结果格式化脚本",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- run_all.py：一键运行脚本",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- requirements.txt：依赖包列表",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- install_dependencies.sh：依赖安装脚本",
                spacing: { after: 200 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "代码执行步骤：", bold: true })
                ],
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "步骤0：安装依赖（首次运行需要）",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "bash install_dependencies.sh",
                spacing: { after: 100 }
            }),
            new Paragraph({
                text: "或手动安装：pip install -r requirements.txt",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "【推荐】一键运行所有步骤：",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "python run_all.py",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "或者分步执行：",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "步骤1：执行 python 01_data_preprocess.py",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "功能：加载并清洗原始拓扑数据，构建图结构",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "输出：生成data文件夹，包含train_data.pkl、test_data.pkl和stats.json",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "步骤2：执行 python 02_feature_extraction.py",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "功能：从图结构中提取节点特征和图特征",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "输出：生成features文件夹，包含train_features.pkl和test_features.pkl",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "步骤3：执行 python 03_model_training.py",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "功能：训练随机森林和梯度提升模型",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "输出：生成models文件夹，包含rootcause_model.pkl和scaler.pkl",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "步骤4：执行 python 04_model_prediction.py",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "功能：使用训练好的模型对测试集进行预测",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "输出：生成results文件夹，包含predictions.pkl和predictions.json",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "步骤5：执行 python 05_format_results.py",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "功能：将预测结果转换为提交格式",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "输出：生成submit文件夹，每个测试案例对应一个.rootcause.json文件",
                spacing: { after: 300 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "注意事项：", bold: true })
                ],
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "1. 所有代码必须在包含train和test文件夹的目录下运行",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "2. 确保Python版本为3.7或以上",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "3. 给出的代码和参数已经是最优参数，无需调参",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "4. 最终提交文件位于submit文件夹中",
                spacing: { after: 300 }
            }),

            new Paragraph({
                text: "4. 预估时长",
                heading: HeadingLevel.HEADING_2,
                spacing: { before: 400, after: 200 }
            }),

            new Paragraph({
                text: "根据服务器性能（CPU 8核16G实例），各步骤预估时长如下：",
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "- 数据预处理流程：耗时约30分钟",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 特征提取流程：耗时约2小时（包含复杂的图算法计算）",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 模型训练流程：耗时约1小时",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 模型预测流程：耗时约30分钟",
                spacing: { after: 50 }
            }),
            new Paragraph({
                text: "- 结果格式化流程：耗时约10分钟",
                spacing: { after: 200 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "总计运行时长：约4-5小时", bold: true })
                ],
                spacing: { after: 200 }
            }),

            new Paragraph({
                text: "注：实际运行时间会根据硬件配置有所差异。GPU加速暂不支持，但不影响最终结果。",
                spacing: { after: 300 }
            }),

            new Paragraph({
                text: "5. 创新点及目前存在问题说明",
                heading: HeadingLevel.HEADING_2,
                spacing: { before: 400, after: 200 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "创新点：", bold: true })
                ],
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "1、多维度特征融合：结合节点的结构特征（度数、邻居关系）、中心性特征（PageRank、介数中心性）和属性特征（设备类型、供应商），构建全面的特征体系。相比单一特征，多维度特征能更准确地刻画节点在网络中的角色和重要性。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "2、集成学习策略：采用随机森林和梯度提升两种互补的集成学习算法，通过模型融合提高预测的稳定性和准确性。随机森林擅长处理高维特征和非线性关系，梯度提升能够精细优化预测效果，两者结合能够充分发挥各自优势。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "3、类别不平衡优化：针对故障节点占比极小的问题，采用类别权重调整和样本权重两种方法，使模型能够有效学习少数类样本的特征。相比传统方法，这种优化策略显著提高了对故障节点的召回率。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "4、动态Top-K预测策略：根据网络规模自适应调整预测的根因节点数量，既避免了预测过多造成的误报，又确保了对复杂故障场景的覆盖。结合概率阈值过滤，进一步提高预测的精准度。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "5、端到端自动化流程：构建了从数据加载到结果输出的完整自动化流水线，各模块间数据流转清晰，便于维护和扩展。一键运行脚本极大简化了使用流程，提高了可操作性。",
                spacing: { after: 300 }
            }),

            new Paragraph({
                children: [
                    new TextRun({ text: "存在问题：", bold: true })
                ],
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "1、计算效率问题：特征提取阶段需要计算大量图算法（如介数中心性、PageRank等），在大规模网络上计算复杂度较高，导致耗时较长。未来可以考虑使用更高效的近似算法或并行计算来优化。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "2、特征表示局限性：当前特征主要基于图结构统计信息，未充分利用时序信息和日志文本内容。如果能够结合时间序列分析和自然语言处理技术，提取更丰富的特征，可能会进一步提升预测效果。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "3、模型可解释性不足：集成学习模型虽然预测准确率较高，但模型的可解释性相对较弱，难以给出明确的决策依据。在实际应用中，可能需要结合规则引擎或可解释AI技术来增强模型的透明度。",
                spacing: { after: 100 }
            }),

            new Paragraph({
                text: "4、泛化能力待验证：模型在训练数据上表现良好，但在不同网络拓扑结构和故障类型上的泛化能力还需要更多测试数据验证。未来可以通过增加更多样化的训练数据和迁移学习技术来提升模型的鲁棒性。",
                spacing: { after: 100 }
            })
        ]
    }]
});

docx.Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("网络故障根因分析团队代码使用文档.docx", buffer);
    console.log("文档创建成功！");
});
