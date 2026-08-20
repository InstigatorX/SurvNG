import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { peopleFilterLabel, peopleModeLabel, peopleObservationRequestPlan, peopleWorkspaceSearch, readPeopleWorkspaceQuery } from "../src/peopleWorkspace.mjs";

assert.deepEqual(readPeopleWorkspaceQuery(""), {
  mode: "review", status: "unknown", cameraId: "", personId: "", page: 0, faceId: "", clusterId: "",
});
assert.deepEqual(readPeopleWorkspaceQuery("?status=suggested&camera=gate&person=4&page=3&face=88"), {
  mode: "review", status: "suggested", cameraId: "gate", personId: "4", page: 2, faceId: "88", clusterId: "",
});
assert.equal(readPeopleWorkspaceQuery("?status=invalid&page=-3").status, "unknown");
assert.deepEqual(readPeopleWorkspaceQuery("?mode=clusters&cluster=12"), {
  mode: "clusters", status: "unknown", cameraId: "", personId: "", page: 0, faceId: "", clusterId: "12",
});
assert.equal(readPeopleWorkspaceQuery("?mode=invalid").mode, "review");
assert.equal(peopleWorkspaceSearch({ status: "known", cameraId: "front-door", page: 1, faceId: 92 }), "?status=known&camera=front-door&page=2&face=92");
assert.equal(peopleWorkspaceSearch(), "");
assert.equal(peopleWorkspaceSearch({ mode: "clusters", clusterId: 4 }), "?mode=clusters&cluster=4");
assert.equal(peopleFilterLabel("unusable"), "Unusable");
assert.equal(peopleModeLabel("people"), "People profiles");
assert.deepEqual(peopleObservationRequestPlan({ mode: "review", pageSize: 24 }), {
  observations: "/api/faces/review/queue?limit=24",
  count: "",
});
assert.deepEqual(peopleObservationRequestPlan({ mode: "clusters", pageSize: 24 }), {
  observations: "",
  count: "",
});
assert.deepEqual(peopleObservationRequestPlan({
  mode: "people", status: "known", cameraId: "gate", personId: "4", page: 2, pageSize: 24,
}), {
  observations: "/api/faces/observations?status=all&limit=24&offset=48&camera_id=gate&person_id=4",
  count: "/api/faces/observations/count?status=all&camera_id=gate&person_id=4",
});

const facesPageSource = readFileSync(new URL("../src/people/FacesPage.jsx", import.meta.url), "utf8");
assert.match(facesPageSource, /const \[peopleLoadError, setPeopleLoadError\] = useState\(""\)/);
assert.match(facesPageSource, /peopleLoadError[\s\S]*?onClick=\{\(\) => void loadPeople\(\)\}>Retry people/);

console.log("people workspace tests passed");
