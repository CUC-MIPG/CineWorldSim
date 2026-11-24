# CineWorldSim

# Event-Centric World-Aware Multi-Agent Cinematic Video Generation Framework

## Project Progress Summary and Future Planning

---

# 1. Project Overview

## 1.1 Project Motivation

近年来，基于大语言模型和视频生成模型的发展推动了自动化影视内容生成的发展。然而，现有 Video Generation Agent 通常采用：

Story / Event
|
v
Prompt Expansion
|
v
Single Video Clip Generation

即将一个叙事事件直接扩展为连续的视频描述。

例如：

Input:

> A man enters a house.

Existing methods:

A cinematic shot of a man walking toward the house.
The camera follows his footsteps.
He takes out the key and opens the door.
The camera follows him entering the room.

最终生成：

One continuous video clip

该范式存在一个核心问题：

## Narrative Event ≠ Single Continuous Shot

在真实影视制作流程中，一个完整事件通常通过多个镜头共同表达：

Event:
Man enters a house

```
|
v
```

Multi-shot Storyboard:
Shot 1:
High angle establishing shot
Man approaches the house
Shot 2:
Tracking shot
Man walks through corridor
Shot 3:
Close-up shot
Man takes out the key
Shot 4:
Interior shot
Man enters the room

因此，影视生成系统需要学习：

Event

↓

Multi-shot Storyboard

↓

Video Sequence

而不是：

Event

↓

Single Video Clip

---



# 2. Research Objective

本项目提出：

## Event-Centric Cinematic Generation Paradigm

核心目标：

构建一个面向电影级内容创作的多智能体视频生成框架，使 AI 能够：

1. 理解故事事件及人物目标；
2. 建立动态世界状态；
3. 将事件自动分解为符合影视语言的多镜头故事板；
4. 基于镜头级规划生成连续视频；
5. 完成后期制作形成完整影视内容。

整体目标：

Narrative Intelligence

```
+
```

World Consistency

```
+
```

Cinematic Shot Planning

```
+
```

Video Generation

```
+
```

# Post Production

AI Director Pipeline

---



# 3. Overall Framework



## CineWorldSim Architecture

```
                 User Story

                      |
                      v


         Narrative Understanding Agent


                      |
                      v


      =================================

          World-aware Narrative Simulation

      =================================


            Character Agent

            World Model Agent

            State Reasoning Agent


                      |
                      v


            Narrative Events


                      |
                      v


      =================================

         Event-to-Multi-shot Director

             (Core Innovation)

      =================================


                      |
                      v


            Cinematic Storyboard


      +---------------+---------------+

      |               |               |

    Shot 1          Shot 2          Shot 3

      |               |               |

      v               v               v


  Camera        Camera          Camera

  Language      Language        Language


      |               |               |

      +---------------+---------------+

                      |

                      v


            Video Generation Agents


                      |

                      v


            Post-production Agent


                      |

                      v


           Final Cinematic Video
```

---



# 4. Existing System Implementation

当前 CineWorldSim 已经完成：

## 4.1 Narrative World Simulation Layer

负责：

> Understanding what happens in the story.

包含：

- OrchestratorAgent
- CharacterAgent
- WorldModelAgent
- StateAnalysisMachine
- WorldDeltaJudge
- WorldStateTracker

当前流程：

Story
↓
Character Creation
↓
World Blueprint Extraction
↓
Narrative Deduction
↓
World State Update
↓
Event Representation

该部分实现故事逻辑和世界连续性建模。

---



# 5. World Modeling Module



## Goal

解决：

Shot-to-shot visual inconsistency

传统方法：

Shot1
Shot2
Shot3
(independent generation)

容易导致：

- Location inconsistency
- Object disappearance
- Character appearance drift

CineWorldSim 引入：

## Dynamic World State

维护：

World State

|
+-- Location

|
+-- Objects

|
+-- Character Status

随着剧情推进：

Event
↓
World Delta
↓
Updated World State
↓
New Shot Generation

---



# 6. Core Innovation Module



# Event-to-Multi-shot Storyboard Agent



## Motivation

Existing methods:

Event
↓
Long Video Prompt
↓
Single Clip

Proposed:

Event  
↓

Long Video Prompt 

↓ 

Single Clip

↓   
Shot Decomposition  
↓  
Storyboard  
↓  
Multi-shot Video

---



# 6.1 Input

Example:

Event:
A man enters a house.

World Context:

Location:
Old house
Character:
John
Object:
Locked door

---



# 6.2 Output

Storyboard:

## Shot 1

Type:
Establishing Shot
Camera:
High angle
Action:
Man approaches the house.
Purpose:
Establish environment.

---



## Shot 2

Type:
Tracking Shot
Camera:
Side tracking movement
Action:
Man walks through corridor.
Purpose:
Show movement.

---



## Shot 3

Type:
Close-up
Camera:
Macro
Action:
Man takes out the key.
Purpose:
Highlight key action.

---



## Shot 4

Type:
Interior Shot
Camera:
Reverse angle
Action:
Man enters the room.
Purpose:
Complete event transition.

---



# 7. Agent Design



## 7.1 Narrative Agent

Input:

Story

Output:

Characters
Events
Goals
Relations

---



## 7.2 World Model Agent

Output:

Location graph
Object states
Spatial relations
Visual references

---



## 7.3 Event Reasoning Agent

Purpose:

Convert narrative steps into atomic events.

Output:

Event:
Man enters house
Precondition:
Door locked
Action:
Unlock door
Postcondition:
Inside house

---



## 7.4 Cinematic Director Agent

(Core Innovation)

Function:

Event
↓
Shot Sequence

Generate:

- Shot number
- Shot type
- Camera angle
- Camera motion
- Composition
- Duration
- Transition

---



## 7.5 Video Generation Agent

Input:

Shot-level prompts
Reference frames
World assets

Output:

Shot videos

---



## 7.6 Post Production Agent

Functions:

- Video assembly
- Audio generation
- Narration
- Music
- Editing

---



# 8. Current Implementation Mapping

Existing Module:


| Module                | Function                |
| --------------------- | ----------------------- |
| OrchestratorAgent     | Story understanding     |
| CharacterAgent        | Character reasoning     |
| WorldModelAgent       | World construction      |
| StateAnalysisMachine  | Narrative simulation    |
| WorldDeltaJudge       | World update detection  |
| WorldStateTracker     | Dynamic world memory    |
| EventKeyframeRenderer | Event visualization     |
| Path Orchestrator     | Narrative path ordering |
| Shot Planner Agent    | Shot generation         |
| VideoAgent            | Video synthesis         |
| Post Production Agent | Media assembly          |


---



# 9. Main Scientific Contributions



## Contribution 1



### Event-Centric Video Generation Paradigm

提出：

Event
↓
Multi-shot Storyboard
↓
Video

替代传统：

Event
↓
Single Clip

---



## Contribution 2



### Cinematic Director Agent

首次显式建模：

Narrative Event
↓
Professional Shot Sequence

学习影视工业中的镜头组织规律。

---



## Contribution 3



### World-aware Multi-shot Planning

结合：

- World state
- Character state
- Object state

保证：

Narrative Consistency

Visual Consistency

---



## Contribution 4



### End-to-End AI Director Workflow

实现：

Story Understanding
↓
Simulation
↓
Storyboard
↓
Generation
↓
Editing

完整模拟真实影视生产流程。

---



# 10. Dataset and Training Plan



## Movie-level Dataset

利用电影数据：

Movie
↓
Scene/Event
↓
Multiple Shots
↓
Video Clips

构造：

## Event-to-Shot Instruction Dataset

Example:

Input:

Event:
A detective discovers a hidden room.

Output:

Shot 1:
Wide shot:
Detective enters abandoned house.
Shot 2:
Medium shot:
Detective observes strange wall.
Shot 3:
Close-up:
Detective opens hidden door.
Shot 4:
POV:
Detective sees hidden room.

训练：

Event-to-Multi-shot Planner

---



# 11. Experimental Plan



## Baselines

Compare:

### Traditional Video Generation

- Text-to-video models



### Agent-based Video Generation

- Existing Video Agents



### Proposed

CineWorldSim

---



# Evaluation



## Narrative Metrics

- Event completeness
- Story coherence
- Character consistency



## Cinematic Metrics

- Shot diversity
- Camera language accuracy
- Editing quality



## Visual Metrics

- Video quality
- Temporal consistency
- Identity consistency

---



# 12. Current Progress

Completed:

[x] Narrative simulation framework

[x] Character agent

[x] World model

[x] World state tracking

[x] Event keyframe generation

[x] Path orchestration

[x] Shot planning framework

[x] Video generation interface

[x] Post-production workflow

---



# 13. Future Development Roadmap



## Phase 1

Improve Event Representation

- Event extraction
- Event graph construction
- Narrative hierarchy modeling

---



## Phase 2

Develop Event-to-Multi-shot Director

Implement:

- Shot decomposition
- Storyboard generation
- Camera planning

---



## Phase 3

Movie Dataset Training

Build:

Event-Shot Pair Dataset

Fine-tune:

Event-to-Shot Planner LLM

---



## Phase 4

End-to-End Cinematic Generation

Integrate:

Event Planning

World Simulation

Storyboard

Video Generation

Post Production

---



# 14. Final Vision

CineWorldSim aims to build:

## An AI Film Director System

Not only generating videos,

but understanding:

What happens?
↓
How should it be filmed?
↓
How should shots be organized?
↓
How should the final movie be produced?

最终实现：

Story
↓
AI Director
↓
Cinematic Video