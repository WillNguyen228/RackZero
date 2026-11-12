"use strict";

import { createDPAgent, createGreedyAgent } from "./agents.js";
import { createDQNAgent } from "./dqn.js";

let dictionary = null;
let dqnModel = null;
const agents = new Map();

self.onmessage = ({ data }) => {
  try {
    if (data.type === "initialize") {
      dictionary = new Set(data.words);
      dqnModel = data.dqnModel;
      agents.clear();
    } else if (data.type === "dqn-model") {
      dqnModel = data.model;
      agents.delete("dqn");
    } else if (data.type === "move") {
      if (!dictionary) throw new Error("Agent is not initialized.");
      if (!agents.has(data.opponent)) {
        if (data.opponent === "greedy") agents.set("greedy", createGreedyAgent(dictionary));
        else if (data.opponent === "dp") agents.set("dp", createDPAgent(dictionary));
        else if (data.opponent === "dqn") agents.set("dqn", createDQNAgent(dictionary, dqnModel));
        else throw new Error("That opponent is not implemented.");
      }
      self.postMessage({ id: data.id, action: agents.get(data.opponent)(data.position) });
    } else {
      throw new Error("Unknown agent request.");
    }
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message });
  }
};
