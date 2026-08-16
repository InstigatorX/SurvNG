import assert from "node:assert/strict";
import { peopleFilterLabel, peopleWorkspaceSearch, readPeopleWorkspaceQuery } from "../src/peopleWorkspace.mjs";

assert.deepEqual(readPeopleWorkspaceQuery(""), {
  status: "unknown", cameraId: "", personId: "", page: 0, faceId: "",
});
assert.deepEqual(readPeopleWorkspaceQuery("?status=suggested&camera=gate&person=4&page=3&face=88"), {
  status: "suggested", cameraId: "gate", personId: "4", page: 2, faceId: "88",
});
assert.equal(readPeopleWorkspaceQuery("?status=invalid&page=-3").status, "unknown");
assert.equal(peopleWorkspaceSearch({ status: "known", cameraId: "front-door", page: 1, faceId: 92 }), "?status=known&camera=front-door&page=2&face=92");
assert.equal(peopleWorkspaceSearch(), "");
assert.equal(peopleFilterLabel("unusable"), "Unusable");

console.log("people workspace tests passed");
