export const generateInitialMindMapData = (keywords) => {
  const nodes = [{ id: "뉴스", group: 0 }];
  const links = [];

  Object.keys(keywords).forEach((category, i) => {
    const groupId = i + 1;
    nodes.push({ id: category, group: groupId });
    links.push({ source: "뉴스", target: category });
  });

  return { nodes, links };
};

export const getSubNodesAndLinks = (category, subKeywords) => {
  const nodes = [];
  const links = [];

  subKeywords.forEach((sub) => {
    const subId = `${sub}`;
    nodes.push({ id: subId, group: category });
    links.push({ source: category, target: subId });
  });

  return { nodes, links };
};
